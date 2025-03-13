import os
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Optional, Type, Union
from multiprocessing import Process, Queue

import torch
import torch.distributed as dist

from iroh_py import create_receiver, create_sender
from logger import get_logger
from serializer import Serializer
from world import get_world

class Request(ABC):
    @abstractmethod
    def wait(self) -> Optional[torch.Tensor]:
        pass

class P2PComm(ABC):
    def __init__(self):
        self.world = get_world()

    @abstractmethod
    def send(self, data: torch.Tensor) -> None:
        pass

    @abstractmethod
    def recv(self) -> torch.Tensor:
        pass

    @abstractmethod
    def irecv(self, tag: int, shape: tuple[int, ...]) -> Request:
        pass

    @abstractmethod
    def destroy(self):
        pass

_COMM: Optional[P2PComm] = None


def setup_comm(target: Type[P2PComm], **kwargs):
    global _COMM
    assert _COMM is None, "Comm already setup"
    _COMM = target(**kwargs)


def get_comm() -> P2PComm:
    global _COMM
    assert _COMM is not None, "Comm not setup"
    return _COMM

class TorchRequest(Request):
    def __init__(self, tensor: torch.Tensor, work: Union[None, int, dist.Work]):
        self.tensor = tensor
        self.work = work

    def wait(self) -> Optional[torch.Tensor]:
        if isinstance(self.work, dist.Work):
            self.work.wait()
        return self.tensor


class TorchP2PComm(P2PComm):
    def __init__(
        self,
        fwd_shape: tuple[int, ...],
        bwd_shape: tuple[int, ...],
        fwd_dtype: torch.dtype,
        bwd_dtype: torch.dtype,
        device: torch.device,
        num_prompt_tokens: int,
        init_method: str = "env://",
        backend: str = "nccl",
    ):
        super().__init__()
        self.logger = get_logger()
        self.fwd_shape, self.bwd_shape = fwd_shape, bwd_shape
        self.fwd_prefill_shape = (fwd_shape[0], num_prompt_tokens, fwd_shape[2])
        self.bwd_prefill_shape = bwd_shape
        self.fwd_dtype, self.bwd_dtype = fwd_dtype, bwd_dtype
        self.device = device
        self.num_prompt_tokens = num_prompt_tokens
        if not os.environ.get("MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = "localhost"
        if not os.environ.get("MASTER_PORT"):
            os.environ["MASTER_PORT"] = "29500"
        self.process_group = dist.init_process_group(
            init_method=init_method, backend=backend
        )

    def _send_fwd(self, send_func: Callable, hidden_states: torch.Tensor, tag: int) -> TorchRequest:
        """Sends hidden states to the next stage"""
        next_rank = self.world.rank + 1
        assert next_rank < self.world.size
        self.logger.debug(f"send_fwd {hidden_states=} to {next_rank=} with {tag=}")
        send_req = send_func(hidden_states, dst=next_rank, tag=tag)
        return TorchRequest(None, send_req)

    def _send_bwd(self, send_func: Callable, next_token: torch.Tensor, tag: int) -> TorchRequest:
        """Sends next token to the first stage"""
        self.logger.debug(
            f"send_bwd {next_token=} to {self.world.first_stage_rank=} with {tag=}"
        )
        send_req = send_func(next_token, dst=self.world.first_stage_rank, tag=tag)
        return TorchRequest(None, send_req)

    def _recv_fwd(self, recv_func: Callable, tag: int, shape: tuple[int, ...]) -> TorchRequest:
        """Receives hidden states from the previous stage"""
        prev_rank = self.world.rank - 1
        assert prev_rank >= 0
        hidden_states = torch.empty(shape, dtype=self.fwd_dtype, device=self.device)
        self.logger.debug(
            f"recv_fwd into {hidden_states=} from {prev_rank=} with {tag=}"
        )
        recv_req = recv_func(hidden_states, src=prev_rank, tag=tag)
        return TorchRequest(hidden_states, recv_req)

    def _recv_bwd(self, recv_func: Callable, tag: int, shape: tuple[int, ...]) -> TorchRequest:
        """Receives next token from the last stage"""
        next_token = torch.empty(shape, dtype=self.bwd_dtype, device=self.device)
        self.logger.debug(
            f"recv_bwd into {next_token=} from {self.world.last_stage_rank=} with {tag=}"
        )
        recv_req = recv_func(next_token, src=self.world.last_stage_rank, tag=tag)
        return TorchRequest(next_token, recv_req)

    def send(self, tensor: torch.Tensor, tag: int) -> None:
        """Sends tensor (hidden states or next token) to the correct next rank"""
        if self.world.size == 1:
            return None
        send_fwd_or_bwd = self._send_bwd if self.world.is_last_stage else self._send_fwd
        send_fwd_or_bwd(dist.send, tensor, tag).wait()

    def recv(self, tag: int, prefill: bool = False) -> Optional[torch.Tensor]:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        if self.world.size == 1:
            return None
        shape = (
            (self.bwd_shape, self.bwd_prefill_shape)
            if self.world.is_first_stage
            else (self.fwd_shape, self.fwd_prefill_shape)
        )
        recv_fwd_or_bwd = self._recv_bwd if self.world.is_first_stage else self._recv_fwd
        return recv_fwd_or_bwd(dist.recv, tag, shape[int(prefill)]).wait()

    def isend(self, tensor: torch.Tensor, tag: int) -> Request:
        """Sends tensor (hidden states or next token) to the correct next rank"""
        send_fwd_or_bwd = self._send_bwd if self.world.is_last_stage else self._send_fwd
        return send_fwd_or_bwd(dist.isend, tensor, tag)

    def irecv(self, tag: int, prefill: bool = False) -> Request:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        shape = (
            (self.bwd_shape, self.bwd_prefill_shape)
            if self.world.is_first_stage
            else (self.fwd_shape, self.fwd_prefill_shape)
        )
        recv_fwd_or_bwd = self._recv_bwd if self.world.is_first_stage else self._recv_fwd
        return recv_fwd_or_bwd(dist.irecv, tag, shape[int(prefill)])

    def destroy(self):
        if dist.is_initialized():
            dist.destroy_process_group()

class IrohRequest(Request):
     def __init__(self, queue: Queue, serializer: Serializer, device: torch.device):
         self.queue = queue
         self.serializer = serializer
         self.device = device

     def wait(self) -> torch.Tensor:
         return self.serializer.deserialize(self.queue.get()).to(self.device)

class IrohP2PComm(P2PComm):
    """IrohNode running receiver and sender in separate processes with non-blocking API"""
    def __init__(self, serializer: Serializer, device: torch.device):
        super().__init__()
        self.serializer = serializer
        self.device = device
        self.logger = get_logger()

        if self.world.size <= 1:
            return

        # Queues for communication
        self.send_queue = Queue()      # Data to send
        self.recv_queue = Queue()      # Received data
        self.send_result_queue = Queue()  # Operation results
        self.id_queue = Queue()        # For node_id
        self.control_queue = Queue()   # For control signals to receiver
        
        # Start processes
        self.receiver = Process(target=self._run_async, args=(self._run_receiver, self.recv_queue, self.id_queue, self.control_queue))
        self.sender = Process(target=self._run_async, args=(self._run_sender, self.send_queue, self.send_result_queue))
        
        self.receiver.start()
        self.sender.start()
        
        # Connect to other node
        self.receiver_id = self.id_queue.get()
        self.logger.info(f"Listening on {self.receiver_id}")

        # Connect to the remote node
        while True:
            try:
                self.logger.info("Please enter the server's public key: ")
                self.connect(input().strip())
                break
            except Exception as e:
                self.logger.error(f"Error connecting to node: {e}")
                continue

        # Wait for connection to be established
        time.sleep(1)
        self.logger.info("Connected!")

    def _run_async(self, func, *args):
        """Helper to run async functions in a process"""
        asyncio.run(func(*args))
        
    async def _receive_with_timeout(self, receiver, timeout):
        """Helper method to receive with timeout"""
        try:
            return await asyncio.wait_for(receiver.recv(), timeout)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            return None

    async def _run_receiver(self, recv_queue: Queue, id_queue: Queue, control_queue: Queue):
        """Process function for the receiver"""
        logger = get_logger()
        logger.debug(f"Running receiver (p={os.getpid()}, pp={os.getppid()})")
        
        # Create receiver
        receiver = create_receiver()
        node_id = receiver.get_node_id()

        # Share node_id with parent process
        id_queue.put(node_id)

        # Wait for connection
        while not receiver.is_ready():
            # Check for early shutdown command
            if not control_queue.empty():
                cmd = control_queue.get_nowait()
                if cmd == "shutdown":
                    print(f"Shutting down receiver before connection (pp={os.getppid()})")
                    receiver.shutdown()
                    return
            await asyncio.sleep(0.01)

        # Continuous receive loop
        running = True
        while running:
            # Check for control commands
            if not control_queue.empty():
                try:
                    cmd = control_queue.get_nowait()
                    if cmd == "shutdown":
                        # print(f"Receiver received shutdown command (pp={os.getppid()})")
                        running = False
                        break
                except:
                    pass  # No commands
            
            # Try to receive data but also check control_queue periodically
            try:
                # Create a receive task
                receive_task = asyncio.create_task(self._receive_with_timeout(receiver, 0.01))

                # Wait a short time to allow for receiving data
                await asyncio.sleep(0.1)
                
                # Check if we received something
                if receive_task.done() and not receive_task.cancelled():
                    try:
                        data = receive_task.result()
                        if data is not None:
                            self.logger.debug(f"Received: {data}")
                            recv_queue.put(data)
                    except Exception as e:
                        if running:  # Only log errors if we're still supposed to be running
                            self.logger.error(f"Receive task error: {e}")
                else:
                    # Cancel the task if it's still running
                    if not receive_task.done():
                        receive_task.cancel()
                        
            except Exception as e:
                if running:  # Only log errors if we're still supposed to be running
                    self.logger.error(f"Receiver error: {e}")
                await asyncio.sleep(0.05)  # Small delay on error
                
        print(f"Shutting down receiver (pp={os.getppid()})")
        receiver.shutdown()

    async def _run_sender(self, send_queue: Queue, result_queue: Queue):
        """Process function for the sender"""
        logger = get_logger()
        logger.debug(f"Running sender (p={os.getpid()}, pp={os.getppid()})")
        
        # Create sender
        sender = create_sender()
        
        # Process commands from the queue
        try:
            while True:
                # Non-blocking queue check
                try:
                    cmd, data = send_queue.get(timeout=0.01)
                except:
                    await asyncio.sleep(0.01)
                    continue
                
                try:
                    if cmd == "connect":
                        peer_id = data
                        logger.debug(f"Connecting to: {peer_id} (pp={os.getppid()})")
                        sender.connect(peer_id)
                        
                        # Wait for connection to be ready
                        while not sender.is_ready():
                            await asyncio.sleep(0.01)
                        logger.debug(f"Sender ready (pp={os.getppid()})")
                        
                        # Signal completion
                        result_queue.put(True)
                        
                    elif cmd == "send":
                        logger.debug("Sending data")
                        await sender.send(data)
                        
                        # Signal completion
                        result_queue.put(True)
                    elif cmd == "shutdown":
                        break
                        
                except Exception as e:
                    logger.error(f"Sender error: {e}")
                    result_queue.put(f"Error: {e}")
                    
        finally:
            logger.debug(f"Shutting down sender (pp={os.getppid()})")
            sender.shutdown()

    def get_node_id(self):
        return self.receiver_id

    def connect(self, peer_id: str):
        self.send_queue.put(("connect", peer_id))

    def recv(self, **kwargs) -> Optional[torch.Tensor]:
        if self.world.size <= 1:
            return None
        return self.irecv().wait()

    def irecv(self, **kwargs) -> IrohRequest:
        if self.world.size <= 1:
            return None
        self.logger.debug("Creating irecv request")
        return IrohRequest(self.recv_queue, self.serializer, self.device)

    def send(self, tensor: torch.Tensor, **kwargs) -> None:
        if self.world.size <= 1:
            return
        data = self.serializer.serialize(tensor)
        self.logger.debug(f"Putting into send queue: {data}")
        self.send_queue.put(("send", data))

    def destroy(self):
        """Shutdown the node and all processes."""
        if self.world.size <= 1:
            return
        self.logger.debug("Shutting down")
        
        # Signal both processes to shutdown
        self.send_queue.put(("shutdown", None))
        self.control_queue.put("shutdown")
        
        # Wait for processes to terminate with a timeout
        self.sender.join(timeout=5)
        self.receiver.join(timeout=5)
        
        # If processes are still alive, terminate them
        if self.sender.is_alive():
            self.logger.warn("Forcibly terminating sender process")
            self.sender.terminate()
            
        if self.receiver.is_alive():
            self.logger.warn("Forcibly terminating receiver process")
            self.receiver.terminate()


# class IrohP2PCommOld(P2PComm):
#     def __init__(self, serializer: Serializer, device: torch.device):
#         super().__init__()
#         self.serializer = serializer
#         self.device = device
#         self.logger = get_logger()
#         self.node = None
#         self.executor = ThreadPoolExecutor(max_workers=2) # one for sending, one for receiving
# 
#         if self.world.size <= 1:
#             return
#         self._setup_iroh()
# 
#     def _setup_iroh(self):
#         # Create iroh node
#         self.node = create_connector()
#         self.logger.info(f"Connect to node with: {self.node.get_node_id()}")
# 
#         # Connect to the remote node
#         while True:
#             try:
#                 self.logger.info("Please enter the server's public key: ")
#                 self.node.connect(input().strip())
#                 break
#             except Exception as e:
#                 self.logger.error(f"Error connecting to node: {e}")
#                 continue
# 
#         # Wait for connection to be established
#         while not self.node.is_ready():
#             time.sleep(1 / 100)
#         self.logger.info("Connected!")
# 
#     def send(self, tensor: torch.Tensor, **kwargs) -> None:
#         if self.world.size == 1:
#             return
#         self.logger.debug(f"send {tensor=}")
#         serialized_data = self.serializer.serialize(tensor)
#         self.node.send(serialized_data)
# 
#     def recv(self, **kwargs) -> torch.Tensor:
#         if self.world.size == 1:
#             return
#         serialized_data = self.node.recv()
#         tensor = self.serializer.deserialize(serialized_data).to(self.device)
#         self.logger.debug(f"recv {tensor=}")
#         return tensor
# 
#     def isend(self, tensor: torch.Tensor, **kwargs) -> Optional[Request]:
#         if self.world.size == 1:
#             return
#         future = self.executor.submit(self.send, tensor, **kwargs)
#         self.logger.debug(f"isend {tensor=}")
#         return IrohRequest(future)
# 
#     def irecv(self, **kwargs) -> Optional[Request]:
#         if self.world.size == 1:
#             return
#         future = self.executor.submit(self.recv, **kwargs)
#         self.logger.debug("irecv")
#         return IrohRequest(future)
# 
#     def destroy(self):
#         if self.node is not None:
#             self.executor.shutdown()
#             self.node.shutdown()