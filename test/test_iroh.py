import os
import time
import pytest
import asyncio
from iroh_py import create_sender, create_receiver
from multiprocessing import Process, Queue, Pipe, Event
from multiprocessing.connection import Connection

@pytest.mark.skip(reason="Debugging")
def test_async_sender_receiver():
    async def run_receiver(q: Queue):
        print(f"Running receiver in {os.getpid()} (p: {os.getppid()})")
        receiver = create_receiver()
        node_id = receiver.get_node_id()
        q.put(node_id)
        print(f"Listening on: {node_id}")
        while not receiver.is_ready():
            time.sleep(0.01)
        print("Receiver ready")
        data = await receiver.recv()
        print(f"Received: {data}")
        assert data == b"hello"
        receiver.shutdown()

    async def run_sender(q: Queue):
        print(f"Running sender in {os.getpid()} (p: {os.getppid()})")
        sender = create_sender()
        peer_id = q.get()
        time.sleep(1) # TODO
        print(f"Connecting to: {peer_id}")
        sender.connect(peer_id)
        while not sender.is_ready():
            time.sleep(0.01)
        print("Sender ready")
        await sender.send(b"hello")
        sender.shutdown()

    print(f"Running test on {os.getpid()}")

    def run_async(func, q: Queue):
        asyncio.run(func(q))

    queue = Queue()
    receiver = Process(target=run_async, args=(run_receiver, queue,))
    sender = Process(target=run_async, args=(run_sender, queue,))
    receiver.start()
    sender.start()
    receiver.join()
    sender.join()

class Future:
    def __init__(self):
        self._ready = Event()
        self._result = None
        self._error = None
        
    def set_result(self, result):
        self._result = result
        self._ready.set()
        
    def set_error(self, error):
        self._error = error
        self._ready.set()
        
    def wait(self):
        self._ready.wait()
        if self._error:
            raise RuntimeError(self._error)
        return self._result

class IrohNode:
    """IrohNode running receiver and sender in separate processes"""
    def __init__(self):
        # Queues to communicate data between subprocesses and main process
        self.send_queue = Queue()
        self.recv_queue = Queue()
        self.id_queue = Queue()
        
        # Start receiver and sender in separate processes
        self.receiver = Process(target=self.run_async, args=(self.run_receiver, self.recv_queue, self.id_queue))
        self.sender = Process(target=self.run_async, args=(self.run_sender, self.send_queue, self.id_queue))
        self.receiver.start()
        self.sender.start()
        self.receiver_id = self.id_queue.get()

    def run_async(self, func, q: Queue, conn: Connection):
        asyncio.run(func(q, conn))

    async def run_receiver(self, recv_queue: Queue, id_queue: Queue):
        print(f"Running receiver (p={os.getpid()}, pp={os.getppid()})")
        # Create receiver
        receiver = create_receiver()
        node_id = receiver.get_node_id()

        # Send node id to parent process
        id_queue.put(node_id)
        print(f"Listening on: {node_id} (pp={os.getppid()})")

        # Wait for sender to connect
        while not receiver.is_ready():
            time.sleep(0.01)
        print(f"Receiver ready (pp={os.getppid()})")

        # Receive data
        try:
            while True:
                recv_queue.get()  # Just a signal to receive
                try:
                    data = await receiver.recv()
                    recv_queue.put(("data", data))  # Send back result
                except Exception as e:
                    recv_queue.put(("error", str(e)))  # Send back error
        finally:
            print(f"Shutting down receiver (pp={os.getppid()})")
            receiver.shutdown()

    async def run_sender(self, send_queue: Queue, id_queue: Queue):
        print(f"Running sender (p={os.getpid()}, pp={os.getppid()})")
        # Create sender
        sender = create_sender()
        peer_id = id_queue.get()
        print(f"Got peer id: {peer_id} (pp={os.getppid()})")
        time.sleep(1) # TODO
        sender.connect(peer_id)
        print(f"Connecting to: {peer_id} (pp={os.getppid()})")
        while not sender.is_ready():
            time.sleep(0.01)
        print(f"Sender ready (pp={os.getppid()})")

        try:
            while True:
                msg_type, data = send_queue.get()  # Receive just the message type and data
                if msg_type == "shutdown":
                    break
                try:
                    await sender.send(data)
                    send_queue.put(("ok", None))  # Signal success
                except Exception as e:
                    send_queue.put(("error", str(e)))  # Signal error
        finally:
            print(f"Shutting down sender (pp={os.getppid()})")
            sender.shutdown()

    def get_node_id(self):
        return self.receiver_id

    def connect(self, peer_id: str):
        self.id_queue.put(peer_id)

    def send(self, data: bytes) -> Future:
        future = Future()
        def _handle_result():
            self.send_queue.put(("send", data))
            result_type, result = self.send_queue.get()
            if result_type == "error":
                future.set_error(result)
            else:
                future.set_result(None)
        
        # Run in current process
        _handle_result()
        return future

    def recv(self) -> Future:
        future = Future()
        def _handle_result():
            self.recv_queue.put("recv")  # Signal we want to receive
            result_type, result = self.recv_queue.get()
            if result_type == "error":
                future.set_error(result)
            else:
                future.set_result(result)
        
        # Run in current process
        _handle_result()
        return future

    def shutdown(self):
        print(f"Shutting down in {os.getpid()}")
        self.send_queue.put(("shutdown", None, None))
        self.receiver.join()
        self.sender.join()

def test_iroh_node():
    def run_node(conn: Connection, rank: int):
        print(f"Running node {rank} in {os.getpid()}")
        node = IrohNode()
        node_id = node.get_node_id()
        conn.send(node_id)
        peer_id = conn.recv()
        node.connect(peer_id)

        time.sleep(5)

        print(f"Node {rank} sending hello")
        node.send(b"hello").wait()
        data = node.recv().wait()
        assert data == b"hello"

        node.shutdown()

    conn_a, conn_b  = Pipe()
    node1 = Process(target=run_node, args=(conn_a, 0))
    node2 = Process(target=run_node, args=(conn_b, 1))
    node1.start()
    node2.start()
    node1.join()
    node2.join()
