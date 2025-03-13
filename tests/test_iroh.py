import os
import time
import pytest
import asyncio
from iroh_py import create_sender, create_receiver, create_node
from multiprocessing import Process, Queue, Pipe
from multiprocessing.connection import Connection

def run_async(func, *args):
    asyncio.run(func(*args))

@pytest.mark.skip(reason="Legacy")
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

    queue = Queue()
    receiver = Process(target=run_async, args=(run_receiver, queue,))
    sender = Process(target=run_async, args=(run_sender, queue,))
    receiver.start()
    sender.start()
    receiver.join()
    sender.join()

def test_async_node():
    async def run_node(conn: Connection, rank: int):
        print(f"Running node (p: {os.getpid()}, pp: {os.getppid()})")
        node = create_node()
        node_id = node.get_node_id()
        time.sleep(1)
        conn.send(node_id)
        peer_id = conn.recv()
        node.connect(peer_id)
        print(f"Connected to: {peer_id} (p: {os.getpid()}, pp: {os.getppid()})")
        time.sleep(1)
        if rank == 1:
            print(f"Sending message (p: {os.getpid()}, pp: {os.getppid()})")
            await node.send(b"hello")
        else:
            data = node.recv()
            print(f"Scheduled recv (p: {os.getpid()}, pp: {os.getppid()})")
            data = await data
            print(f"Received message (p: {os.getpid()}, pp: {os.getppid()})")
            assert data == b"hello"
        node.shutdown()

    conn1, conn2 = Pipe()
    sender = Process(target=run_async, args=(run_node, conn1, 1))
    receiver = Process(target=run_async, args=(run_node, conn2, 2))
    sender.start()
    receiver.start()
    sender.join()
    receiver.join()
