import os
import time
import asyncio
from iroh_py import create_sender, create_receiver
from multiprocessing import Process, Queue, Pipe, Event
from multiprocessing.connection import Connection
from typing import Optional, Any

class IrohFuture:
    """
    A simple Future-like object for non-blocking operations.
    """
    def __init__(self, queue: Queue):
        self.queue = queue
        self.result = None
        self._ready = False
        
    def wait(self):
        """Block until the result is available and return it."""
        if not self._ready:
            self.result = self.queue.get()
            self._ready = True
        return self.result
        
    def ready(self) -> bool:
        """Check if the result is ready without blocking."""
        if self._ready:
            return True
            
        if not self.queue.empty():
            try:
                self.result = self.queue.get_nowait()
                self._ready = True
                return True
            except:
                pass
                
        return False


class AsyncIrohNode:
    """IrohNode running receiver and sender in separate processes with non-blocking API"""
    def __init__(self):
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
        
        # Get the receiver's node ID
        self.receiver_id = self.id_queue.get()

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
        print(f"Running receiver (p={os.getpid()}, pp={os.getppid()})")
        
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
                receive_task = asyncio.create_task(self._receive_with_timeout(receiver, 0.1))
                
                # Wait a short time to allow for receiving data
                await asyncio.sleep(0.1)
                
                # Check if we received something
                if receive_task.done() and not receive_task.cancelled():
                    try:
                        data = receive_task.result()
                        if data is not None:
                            # print(f"Received: {data}")
                            recv_queue.put(data)
                    except Exception as e:
                        if running:  # Only log errors if we're still supposed to be running
                            pass
                            # print(f"Receive task error: {e}")
                else:
                    # Cancel the task if it's still running
                    if not receive_task.done():
                        receive_task.cancel()
                        
            except Exception as e:
                if running:  # Only log errors if we're still supposed to be running
                    pass
                    # print(f"Receiver error: {e}")
                await asyncio.sleep(0.05)  # Small delay on error
                
        print(f"Shutting down receiver (pp={os.getppid()})")
        receiver.shutdown()

    async def _run_sender(self, send_queue: Queue, result_queue: Queue):
        """Process function for the sender"""
        print(f"Running sender (p={os.getpid()}, pp={os.getppid()})")
        
        # Create sender
        sender = create_sender()
        
        # Process commands from the queue
        try:
            while True:
                # Non-blocking queue check
                try:
                    cmd, data = send_queue.get(timeout=0.1)
                except:
                    await asyncio.sleep(0.01)
                    continue
                
                try:
                    if cmd == "connect":
                        peer_id = data
                        # print(f"Connecting to: {peer_id} (pp={os.getppid()})")
                        sender.connect(peer_id)
                        
                        # Wait for connection to be ready
                        while not sender.is_ready():
                            await asyncio.sleep(0.01)
                        # print(f"Sender ready (pp={os.getppid()})")
                        
                        # Signal completion
                        result_queue.put(True)
                        
                    elif cmd == "send":
                        # print(f"Sending data: {data}")
                        await sender.send(data)
                        # print(f"Sent data: {data}")
                        
                        # Signal completion
                        result_queue.put(True)
                        
                    elif cmd == "shutdown":
                        break
                        
                except Exception as e:
                    print(f"Sender error: {e}")
                    result_queue.put(f"Error: {e}")
                    
        finally:
            print(f"Shutting down sender (pp={os.getppid()})")
            sender.shutdown()

    def get_node_id(self):
        """Get the node ID of the receiver."""
        return self.receiver_id

    def connect(self, peer_id: str):
        """
        Connect to a peer. Returns a future-like object.
        """
        self.send_queue.put(("connect", peer_id))
        return IrohFuture(self.send_result_queue)

    def send(self, data: bytes):
        """
        Send data to a connected peer. Returns a future-like object.
        """
        self.send_queue.put(("send", data))
        return IrohFuture(self.send_result_queue)

    def recv(self):
        """
        Receive data from a peer. Returns a future-like object.
        """
        # Simply return a future that will get the next received message
        return IrohFuture(self.recv_queue)

    def shutdown(self):
        """Shutdown the node and all processes."""
        # print(f"Shutting down in {os.getpid()}")
        
        # Signal both processes to shutdown
        self.send_queue.put(("shutdown", None))
        self.control_queue.put("shutdown")
        
        # Wait for processes to terminate with a timeout
        self.sender.join(timeout=5)
        self.receiver.join(timeout=5)
        
        # If processes are still alive, terminate them
        if self.sender.is_alive():
            # print("Forcibly terminating sender process")
            self.sender.terminate()
            
        if self.receiver.is_alive():
            # print("Forcibly terminating receiver process")
            self.receiver.terminate()

def work(rank: int, step: int, idx: str):
    current_time = time.time()
    minutes, seconds = divmod(current_time, 60)
    print(f"Node {rank}: Step {step}: Forward {idx} [{int(minutes):02d}:{int(seconds):02d}]")
    time.sleep(1)

def test_async_iroh_node():
    """Example test function for the AsyncIrohNode."""
    def run_node(conn: Connection, rank: int):
        print(f"Running node {rank} in {os.getpid()}")
        node = AsyncIrohNode()
        node_id = node.get_node_id()
        conn.send(node_id)
        peer_id = conn.recv()
        time.sleep(1)
        
        # Non-blocking connect
        node.connect(peer_id).wait()
        print(f"Node {rank}: Connected to {peer_id}")

        # Non-blocking send
        num_micro_batches = 2
        steps = 5
        if rank == 0:
            # Fill pipeline
            recvs = []
            for i in range(1, num_micro_batches+1):
                work(rank, 1, str(i))
                node.send(f"{str(i)}".encode()).wait()
                recvs.append(node.recv())

            for step in range(2, steps+1):
                for i in range(num_micro_batches):
                    recv = recvs[i].wait().decode()
                    work(rank, step, recv)
                    node.send(f"{str(i)}".encode()).wait()
                    recvs[i] = node.recv()
        
        # Non-blocking receive
        if rank == 1:
            for step in range(1, steps+1):
                for i in range(num_micro_batches):
                    recv = node.recv().wait().decode()
                    work(rank, step, recv)
                    node.send(f"{str(i)}".encode()).wait()
        
        node.shutdown()

    conn_a, conn_b = Pipe()
    node1 = Process(target=run_node, args=(conn_a, 0))
    node2 = Process(target=run_node, args=(conn_b, 1))
    node1.start()
    node2.start()
    node1.join()
    node2.join()


if __name__ == "__main__":
    test_async_iroh_node()