
import socket
import threading
import pickle
import struct

from torch import Tensor



class P2PSockets:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self.connections : dict[int, socket.socket] = {}  # rank -> socket
        self.connections_lock = threading.Lock()

        # Create server socket
        print(f"[{self.host}:{self.port}] Creating server socket")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(5)

    def get_or_create_connection(self, host: str, port: int) -> socket.socket:
        """Get existing connection or create new one to target rank"""
        with self.connections_lock:
            if (host, port) not in self.connections:
                try:
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.connect((host, port))
                    self.connections[(host, port)] = client
                    print(f"[{self.host}:{self.port}] Created new connection to {host}:{port}")
                except Exception as e:
                    print(f"[{self.host}:{self.port}] Failed to connect to {host}:{port}: {e}")
                    raise
            return self.connections[(host, port)]

    def remove_connection(self, host: str, port: int):
        """Remove and close a connection"""
        with self.connections_lock:
            if (host, port) in self.connections:
                try:
                    self.connections[(host, port)].close()
                except:
                    pass
                del self.connections[(host, port)]
                print(f"[{self.host}:{self.port}] Removed connection to {host}:{port}")

    def send_tensor(self, tensor: Tensor, host: str, port: int) -> bool:
        """Send tensor to target rank using persistent connection"""
        try:
            conn = self.get_or_create_connection(host, port)
            buffer = pickle.dumps(tensor)
            size = len(buffer)
            conn.sendall(struct.pack("!Q", size))
            conn.sendall(buffer)
            print(f"[{self.host}:{self.port}] Sent tensor with values {tensor[:5]} to {host}:{port}")
            return True
        except Exception as e:
            print(f"[{self.host}:{self.port}] Error sending to {host}:{port}: {e}")
            self.remove_connection(host, port)
            return False

    def recv_tensor(self, host: str, port: int) -> Tensor:
        """Receive a tensor from the source rank"""
        # Accept connection if we don't have one yet
        if (host, port) not in self.connections:
            conn, addr = self.server.accept()
            with self.connections_lock:
                self.connections[(host, port)] = conn
                print(f"[{self.host}:{self.port}] Accepted connection from {host}:{port}")

        conn = self.connections[(host, port)]
        try:
            # Receive size of incoming tensor
            size_data = conn.recv(8)
            if not size_data:
                raise ConnectionError("Connection closed by peer")
            
            size = struct.unpack("!Q", size_data)[0]
            
            # Receive tensor data
            buffer = bytearray()
            while len(buffer) < size:
                chunk = conn.recv(size - len(buffer))
                if not chunk:
                    raise ConnectionError("Connection closed while receiving")
                buffer.extend(chunk)
            
            # Deserialize tensor
            tensor = pickle.loads(buffer)
            print(f"[{self.host}:{self.port}] Received tensor with values {tensor[:5]} from {host}:{port}")
            return tensor

        except Exception as e:
            print(f"[{self.host}:{self.port}] Error receiving from {host}:{port}: {e}")
            self.remove_connection(host, port)
            raise

    def cleanup(self):
        """Clean up all connections"""
        # Close all outgoing connections
        with self.connections_lock:
            for conn in self.connections.values():
                try:
                    conn.close()
                except:
                    pass
            self.connections.clear()
        
        # Close server socket
        try:
            self.server.close()
        except:
            pass

if __name__ == "__main__":
    import time
    import torch
    import multiprocessing as mp
    import signal
    import sys

    batch_size = 32
    seq_len = 512
    world_size = 3
    host = ["localhost"] * world_size
    port = [29500 + rank for rank in range(world_size)]

    rank2addr = {rank: (host[rank], port[rank]) for rank in range(world_size)}
    print(rank2addr)

    def worker(host: str, port: int, rank: int):
        assert rank < world_size
        p2p = P2PSockets(host=host, port=port)
        time.sleep(1) # wait for other processes to start
        start = True
        try:
            while True:
                # Receive tensor from previous rank (or start from pre-fill)
                if rank == 0 and start:
                    tensor = torch.ones((batch_size, seq_len))
                    start = False
                else:
                    dst_host, dst_port = rank2addr[(rank - 1) % world_size]
                    tensor = p2p.recv_tensor(dst_host, dst_port)
                
                time.sleep(0.5) # Simulate some work
                if rank == 0: # Enumerate inference steps
                    tensor += 1

                # Send tensor to next rank
                src_host, src_port = rank2addr[(rank + 1) % world_size]
                p2p.send_tensor(tensor, src_host, src_port)
        finally:
            p2p.cleanup()

    processes = [mp.Process(target=worker, args=(host[rank], port[rank], rank)) for rank in range(world_size)]

    def signal_handler(sig, frame):
        print("\nShutting down workers...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for p in processes:
        p.start()
    for p in processes:
        p.join()