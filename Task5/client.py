import socket
from pathlib import Path
from recognizer import Signal
import numpy as np
import server
import time
from tqdm import tqdm

HOST = "127.0.0.1"
PORT = 65432

server_socket = socket.socket()


def split(signal: Signal, chuncks_n: int):
    """Yield successive n-sized chunks from l-st"""

    for i in range(0, len(signal.wave), chuncks_n):
        yield signal.wave[i : i + chuncks_n]


audio = Signal.load(Path(r'Task5/Data/test1.wav'))
data = input('Enter:')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.connect((HOST, PORT))
    print(f'Connected to {HOST}:{PORT}')

    server.send_msg(server_socket, str(audio.sample_rate).encode())

    _len = len(audio.wave)
    iterable = tqdm(split(audio, 1000), total=_len // 1000 + np.sign(_len % 1000))
    for chunk in iterable:
        packet = np.array(chunk, dtype=np.float32).tobytes()
        server.send_msg(server_socket, packet)
        time.sleep(0.2)

    server.send_msg(server_socket, server.END_MESSAGE.encode())

    msg = server.recv_msg(server_socket)
    if msg:
        print('Result: ', msg.decode())

    server_socket.close()
