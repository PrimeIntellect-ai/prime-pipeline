import time
import asyncio
from iroh_py import create_sender, create_receiver
from multiprocessing import Process, Queue
import threading
import functools

class AwaitToWait:
    """
    A wrapper that converts an awaitable to a blocking operation with a .wait() method
    that works in synchronous code without a running event loop.
    """
    def __init__(self, coro_fn, *args, **kwargs):
        """
        Initialize with a coroutine factory function and its arguments
        rather than a coroutine directly.
        
        Args:
            coro_fn: A function that returns a coroutine when called
            *args, **kwargs: Arguments to pass to coro_fn
        """
        self._coro_fn = coro_fn
        self._args = args
        self._kwargs = kwargs
        self._result = None
        self._exception = None
        self._done_event = threading.Event()
        
        # Start the background task immediately
        self._thread = threading.Thread(target=self._run_in_thread)
        self._thread.daemon = True
        self._thread.start()
    
    def _run_in_thread(self):
        """Run the coroutine function in a new thread with its own event loop"""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create the coroutine in this thread where we have an event loop
            coro = self._coro_fn(*self._args, **self._kwargs)
            self._result = loop.run_until_complete(coro)
        except Exception as e:
            self._exception = e
        finally:
            loop.close()
            self._done_event.set()
    
    def wait(self):
        """Block until the coroutine completes and return the result"""
        self._done_event.wait()
        if self._exception:
            raise self._exception
        return self._result


def async_wait(async_fn):
    """
    Decorator that converts an async method to one that returns an AwaitToWait object.
    This allows the method to be called from synchronous code.
    
    Example:
        @async_wait
        async def my_async_method(self, arg1, arg2):
            # async implementation
    
    Usage:
        result = obj.my_async_method(arg1, arg2).wait()
    """
    @functools.wraps(async_fn)
    def wrapper(self, *args, **kwargs):
        # Create a function that will capture self and the arguments
        def create_coro():
            return async_fn(self, *args, **kwargs)
        
        return AwaitToWait(create_coro)
    
    return wrapper


class EnhancedReceiver:
    """Wrapper around the Iroh receiver that provides both async and sync interfaces"""
    def __init__(self):
        self._receiver = create_receiver()
    
    def get_node_id(self):
        return self._receiver.get_node_id()
    
    def is_ready(self):
        return self._receiver.is_ready()
    
    async def recv(self):
        return await self._receiver.recv()
    
    @async_wait
    async def recv_wait(self):
        return await self._receiver.recv()
    
    def shutdown(self):
        return self._receiver.shutdown()


class EnhancedSender:
    """Wrapper around the Iroh sender that provides both async and sync interfaces"""
    def __init__(self):
        self._sender = create_sender()
    
    def is_ready(self):
        return self._sender.is_ready()
    
    def connect(self, peer_id):
        return self._sender.connect(peer_id)
    
    async def send(self, data):
        return await self._sender.send(data)
    
    @async_wait
    async def send_wait(self, data):
        return await self._sender.send(data)
    
    def shutdown(self):
        return self._sender.shutdown()


# Example usage in sync code
def sync_run_receiver(queue, rank):
    print(f"[Receiver {rank}] Starting in sync mode")
    receiver = EnhancedReceiver()
    node_id = receiver.get_node_id()
    queue.put(node_id)
    print(f"[Receiver {rank}] Node ID: {node_id}")
    
    # Wait for connection to be ready
    while not receiver.is_ready():
        time.sleep(0.1)

    # Use the wait() pattern in synchronous code
    print(f"[Receiver {rank}] Waiting for message...")
    future = receiver.recv_wait()
    print(f"[Receiver {rank}] Doing some other work...")
    seconds = 3
    while seconds > 0:
        print(f"[Receiver {rank}] Working... {seconds} seconds left")
        time.sleep(1)
        seconds -= 1
    print(f"[Receiver {rank}] Done doing other work")
    result = future.wait()
    print(f"[Receiver {rank}] Got message: {result}")
    
    # Clean shutdown
    receiver.shutdown()
    print(f"[Receiver {rank}] Shut down")

async def async_run_receiver(queue, rank):
    print(f"[Receiver {rank}] Starting in async mode")
    receiver = EnhancedReceiver()
    node_id = receiver.get_node_id()
    queue.put(node_id)
    print(f"[Receiver {rank}] Node ID: {node_id}")
    
    # Wait for connection to be ready
    while not receiver.is_ready():
        await asyncio.sleep(0.1)
    
    # Use the wait() pattern in asynchronous code
    print(f"[Receiver {rank}] Waiting for message...")
    data = await receiver.recv()
    print(f"[Receiver {rank}] Doing some other work...")
    seconds = 3
    while seconds > 0:
        print(f"[Receiver {rank}] Working... {seconds} seconds left")
        time.sleep(1)
        seconds -= 1
    print(f"[Receiver {rank}] Got message: {data}")

    receiver.shutdown()
    print(f"[Receiver {rank}] Shut down")

# Example usage in async code
async def async_run_sender(queue, rank):
    print(f"[Sender {rank}] Starting in async mode")
    sender = EnhancedSender()
    time.sleep(1)  # Give receiver time to start
    
    peer_id = queue.get()
    print(f"[Sender {rank}] Connecting to {peer_id}")
    
    # Connect to peer
    sender.connect(peer_id)
    
    # Wait for connection to be ready
    while not sender.is_ready():
        time.sleep(0.1)
    
    # Send a message
    message = b"Hello from async sender!"
    await sender.send(message)
    print(f"[Sender {rank}] Sent message: {message}")
    
    # Clean shutdown
    sender.shutdown()
    print(f"[Sender {rank}] Shut down")


def run_sender(queue, rank):
    print(f"[Process {rank}] Starting sender process")
    asyncio.run(async_run_sender(queue, rank))


def run_receiver(queue, rank):
    print(f"[Process {rank}] Starting receiver process")
    sync_run_receiver(queue, rank)
    #  asyncio.run(async_run_receiver(queue, rank))


def main():
    queue = Queue()
    sender = Process(target=run_sender, args=(queue, 0))
    receiver = Process(target=run_receiver, args=(queue, 1))
    
    receiver.start()
    sender.start()
    
    receiver.join()
    sender.join()
    
    print("All processes completed")


if __name__ == "__main__":
    main()