import time
import torch
import torch.distributed as dist
import multiprocessing as mp
from functools import partial
from lovely_tensors import monkey_patch; monkey_patch()

next_token = 0

def init_model(rank, world_size):
    def run_forward(x, micro_batch_idx, rank, world_size):
        global next_token
        time.sleep(1)
        if rank != world_size - 1:
            output = torch.full((x.size(0), 1, 4096), x.float().mean(), dtype=torch.long)
        else:
            next_token += 1
            output = torch.full((x.size(0), 1), next_token, dtype=torch.long)
        print(f"[{rank}] Ran forward for micro batch {micro_batch_idx} and got {output=}")
        return output
    return partial(run_forward, rank=rank, world_size=world_size)

def pipeline1(model, batched_tokens, rank, world_size, num_new_tokens):
    """Decodes micro-batches one-by-one ≈ O(num_micro_batches * num_new_tokens * num_devices)"""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 4096)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu
    print(f"[{rank}] {tokens_shape=}, {hidden_states_shape=}, {dtype=}, {device=}")

    if rank == 0: # first stage
        for _ in range(num_new_tokens): # decoding steps
            for i in range(len(batched_tokens)):
                hidden_states = model(batched_tokens[i], i)
                dist.send(hidden_states, dst=1, tag=i)
                next_tokens = torch.empty(tokens_shape, dtype=dtype, device=device)
                dist.recv(next_tokens, src=1, tag=i)
                batched_tokens[i] = torch.cat((batched_tokens[i], next_tokens), dim=1)
    elif rank == world_size - 1: # last stage
        for _ in range(num_new_tokens): # decoding steps
            for i in range(len(batched_tokens)):
                hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                dist.recv(hidden_states, src=0, tag=i)
                next_tokens = model(hidden_states, i)
                dist.send(next_tokens, dst=0, tag=i)
                batched_tokens[i] = torch.cat((batched_tokens[i], next_tokens), dim=1)
    else:
        raise NotImplementedError
    
    return batched_tokens


def pipeline2(model, batched_tokens, rank, world_size, num_new_tokens):
    """Forwards all micro-batches at once, but waits for all micro-batches to be ready"""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 4096)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu
    print(f"[{rank}] {tokens_shape=}, {hidden_states_shape=}, {dtype=}, {device=}")
    
    if rank == 0: # first stage
        for _ in range(num_new_tokens): # decoding steps
            for i in range(len(batched_tokens)):
                hidden_states = model(batched_tokens[i], i)
                dist.send(hidden_states, dst=1, tag=i)
            
            # Now receive all results
            for i in range(len(batched_tokens)):
                next_tokens = torch.empty(tokens_shape, dtype=dtype, device=device)
                dist.recv(next_tokens, src=1, tag=i)
                batched_tokens[i] = torch.cat((batched_tokens[i], next_tokens), dim=1)
    
    elif rank == world_size - 1: # last stage
        for _ in range(num_new_tokens): # decoding steps
            all_hidden_states = []
            for i in range(len(batched_tokens)):
                hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                dist.recv(hidden_states, src=0, tag=i)
                all_hidden_states.append(hidden_states)
            for i in range(len(batched_tokens)):
                next_tokens = model(all_hidden_states[i], i)
                dist.send(next_tokens, dst=0, tag=i)
                batched_tokens[i] = torch.cat((batched_tokens[i], next_tokens), dim=1)
    else:
        raise NotImplementedError
    
    return batched_tokens

def pipeline3(model, batched_tokens, rank, world_size, num_new_tokens):
    """Asynchronously forwards micro-batches and receives results as they come in"""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 4096)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu
    print(f"[{rank}] {tokens_shape=}, {hidden_states_shape=}, {dtype=}, {device=}")
    num_micro_batches = len(batched_tokens)
    
    if rank == 0: # first stage
        # Single set of buffers we'll reuse
        send_reqs = [None] * num_micro_batches     # Store send requests
        recv_buffers = [None] * num_micro_batches  # Store tensors being received
        recv_reqs = [None] * num_micro_batches     # Store receive requests
        
        # Initial forwards for all micro-batches to start the pipeline
        if num_new_tokens > 0:
            for i in range(num_micro_batches):
                hidden_states = model(batched_tokens[i], i)
                send_reqs[i] = dist.isend(hidden_states, dst=1, tag=i)
            
                recv_buffers[i] = torch.empty(tokens_shape, dtype=dtype, device=device)
                recv_reqs[i] = dist.irecv(recv_buffers[i], src=1, tag=i)
        
        # Process remaining tokens while keeping pipeline full
        for token_idx in range(num_new_tokens):
            # Wait for oldest results and update
            for i in range(num_micro_batches):
                recv_reqs[i].wait()
                batched_tokens[i] = torch.cat((batched_tokens[i], recv_buffers[i]), dim=1)
                
                # Immediately process and send next token, except on last iteration
                if token_idx < num_new_tokens - 1:
                    hidden_states = model(batched_tokens[i], i)
                    send_reqs[i] = dist.isend(hidden_states, dst=1, tag=i)
                    
                    recv_buffers[i] = torch.empty(tokens_shape, dtype=dtype, device=device)
                    recv_reqs[i] = dist.irecv(recv_buffers[i], src=1, tag=i)
            
            # Ensure sends complete before next iteration
            for req in send_reqs:
                if not req.is_completed():
                    req.wait()
    
    elif rank == world_size - 1: # last stage
        recv_buffers = [None] * num_micro_batches
        recv_reqs = [None] * num_micro_batches
        send_reqs = [None] * num_micro_batches
        
        # Initialize receives for all micro-batches
        if num_new_tokens > 0:
            for i in range(num_micro_batches):
                recv_buffers[i] = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                recv_reqs[i] = dist.irecv(recv_buffers[i], src=0, tag=i)
        
        # Process tokens while keeping pipeline full
        for token_idx in range(num_new_tokens):
            for i in range(num_micro_batches):
                # Wait for input
                recv_reqs[i].wait()
                
                # Process and send back
                next_tokens = model(recv_buffers[i], i)
                send_reqs[i] = dist.isend(next_tokens, dst=0, tag=i)
                batched_tokens[i] = torch.cat((batched_tokens[i], next_tokens), dim=1)
                
                # Immediately post next receive, except on last iteration
                if token_idx < num_new_tokens - 1:
                    recv_buffers[i] = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                    recv_reqs[i] = dist.irecv(recv_buffers[i], src=0, tag=i)
            
            # Ensure sends complete before next iteration
            for req in send_reqs:
                if not req.is_completed():
                    req.wait()
    
    else:
        raise NotImplementedError
    
    return batched_tokens

def main(rank: int, world_size: int):
    print(f"[{rank}] Running")
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12345", rank=rank, world_size=world_size)

    # Prepare dummy tokens
    batch_size = 2
    prompt_tokens = 1
    tokens = torch.zeros(batch_size, prompt_tokens, dtype=torch.long)

    # Split tokens into micro-batches
    num_micro_batches = world_size
    micro_batch_size = batch_size // num_micro_batches
    micro_batches = list(tokens.split(micro_batch_size, dim=0))

    # Initialize model
    model = init_model(rank, world_size)

    # Run pipelined decoding
    num_new_tokens = 5
    start_time = time.time()
    batched_tokens = pipeline3(model, micro_batches, rank, world_size, num_new_tokens)
    print(f"[{rank}] Time taken: {time.time() - start_time} seconds")
    print(torch.cat(batched_tokens, dim=0))

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    processes = [mp.Process(target=main, args=(rank, world_size)) for rank in range(world_size)]
        
    for p in processes:
        p.start()
    for p in processes:
        p.join()