import os
import time
import pytest
from iroh_py import create_connector, create_sender, create_receiver
from multiprocessing import Process, Queue, Pipe, Event
from multiprocessing.connection import Connection

@pytest.mark.skip(reason="Legacy test")
def test_bi():
    def run_node(conn: Connection):
        print(f"running node in {os.getpid()} (p: {os.getppid()})")
        node = create_connector()
        node_id = node.get_node_id()
        conn.send(node_id)
        time.sleep(1)
        print(f"listening on: {node_id}")
        peer_id = conn.recv()
        node.connect(peer_id)
        while not node.is_ready():
            time.sleep(0.01)
        print(f"connected to: {peer_id}")
        print(f"send/ recv ready: {peer_id}")
        node.shutdown()

    print(f"running test on {os.getpid()}")
    a, b = Pipe()
    node1 = Process(target=run_node, args=(a,))
    node2 = Process(target=run_node, args=(b,))
    node1.start()
    node2.start()
    node1.join()
    node2.join()

@pytest.mark.skip(reason="This test is not working")
def test_uni():
    def run_receiver(q: Queue):
        print(f"running receiver in {os.getpid()} (p: {os.getppid()})")
        receiver = create_receiver()
        node_id = receiver.get_node_id()
        q.put(node_id)
        print(f"listening on: {node_id}")
        while not receiver.is_ready():
            time.sleep(0.01)
        print("receiver ready")
        data = receiver.recv()
        print(f"received: {data}")
        assert data == b"hello"
        receiver.shutdown()

    def run_sender(q: Queue):
        print(f"running sender in {os.getpid()} (p: {os.getppid()})")
        sender = create_sender()
        peer_id = q.get()
        time.sleep(1) # TODO: wait for something explicit here (receiver was not yet ready)
        print(f"connecting to: {peer_id}")
        sender.connect(peer_id)
        while not sender.is_ready():
            time.sleep(0.01)
        print("sender ready")
        sender.send(b"hello")
        sender.shutdown()

    print(f"running test on {os.getpid()}")
    queue = Queue()
    receiver = Process(target=run_receiver, args=(queue,))
    sender = Process(target=run_sender, args=(queue,))
    receiver.start()
    sender.start()
    receiver.join()
    sender.join()

class IrohReceiverSender:
    """IrohNode runs receiver and sender in separate processes"""
    def __init__(self, send_queue: Queue, recv_queue: Queue, conn: Connection):
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.conn = conn
        self.stop = Event()
        
        # Queues to communicate node id subprocesses and main process
        self.send_id, self.recv_id = Queue(), Queue()

        # Start receiver and sender in separate processes
        self.receiver = Process(target=self.run_receiver, args=(recv_queue, self.recv_id))
        self.sender = Process(target=self.run_sender, args=(send_queue, self.send_id))
        self.receiver.start()
        self.sender.start()
        self.receiver_id = self.recv_id.get()

    def run_receiver(self, recv_queue: Queue, recv_id: Queue):
        print(f"Run receiver in {os.getpid()} (p: {os.getppid()})")
        # Create receiver
        receiver = create_receiver()
        node_id = receiver.get_node_id()

        # Send node id to parent process
        recv_id.put(node_id)
        print(f"receiver node id: {node_id} (p={os.getpid()}, pp={os.getppid()})")

        # Wait for sender to connect
        while not receiver.is_ready():
            time.sleep(0.01)
        print(f"receiver ready (p={os.getpid()}, pp={os.getppid()})")

        while True:
            try:
                # Check for stop signal
                if self.stop.is_set():
                    break
                    
                # Blocking receive
                print(f"receiving data: (p={os.getpid()}, pp={os.getppid()})")
                data = receiver.recv()
                print(f"received data: {data} (p={os.getpid()}, pp={os.getppid()})")
                if data:
                    recv_queue.put(data)
            except Exception as e:
                print(f"Receiver error: {e}")
                break

        print(f"receiver shutdown (p={os.getpid()}, pp={os.getppid()})")
        receiver.shutdown()

    def run_sender(self, send_queue: Queue, send_id: Queue):
        print(f"run sender (p={os.getpid()}, pp={os.getppid()})")

        # Create sender
        sender = create_sender()
        peer_id = send_id.get()
        print(f"got peer id: {peer_id} (p={os.getpid()}, pp={os.getppid()})")
        time.sleep(1) # TODO: fix this hardcoded sleep
        sender.connect(peer_id)
        print(f"connecting to: {peer_id} (p={os.getpid()}, pp={os.getppid()})")
        while not sender.is_ready():
            time.sleep(0.01)
        print(f"sender ready (p={os.getpid()}, pp={os.getppid()})")
        while True:
            try:
                # Check for shutdown signal
                if self.stop.is_set() and send_queue.empty():
                    break
                    
                # Non-blocking check for data to send
                try:
                    data = send_queue.get_nowait()
                    print(f"sending data: {data} (p={os.getpid()}, pp={os.getppid()})")
                    sender.send(data)
                except:
                    time.sleep(0.01)  # Prevent busy waiting
            except Exception as e:
                print(f"Sender error: {e}")
                break

        print(f"sender shutdown (p={os.getpid()}, pp={os.getppid()})")
        sender.shutdown()

    def get_node_id(self):
        return self.receiver_id

    def connect(self, peer_id: str):
        self.send_id.put(peer_id)

    def send(self, data: bytes):
        self.send_queue.put(data)

    def recv(self):
        return self.recv_queue.get()

    def shutdown(self):
        print(f"shutdown in {os.getpid()}")
        self.stop.set()
        self.receiver.join()
        self.sender.join()

def test_bi_with_process():
    def run_node(conn: Connection, rank: int):
        print(f"running node {rank} in {os.getpid()}")
        send_queue = Queue()
        recv_queue = Queue()
        node = IrohReceiverSender(send_queue, recv_queue, conn)
        node_id = node.get_node_id()
        conn.send(node_id)
        peer_id = conn.recv()
        node.connect(peer_id)

        time.sleep(3)

        if rank == 0:
            node.send(b"hello")
            node.send(b"world")
        else:
            data = node.recv()
            assert data == b"hello"
            data = node.recv()
            assert data == b"world"
        print("calling node shutdown")
        node.shutdown()

    a, b  = Pipe()
    node1 = Process(target=run_node, args=(a, 0))
    node2 = Process(target=run_node, args=(b, 1))
    node1.start()
    node2.start()
    node1.join()
    node2.join()
