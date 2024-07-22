import socket
from pathlib import Path
from recogniser import Signal
import numpy as np

HOST = "127.0.0.1"
PORT = 65432
DISCONNECT_MESSAGE = np.array([999], dtype=np.float32)

s = socket.socket()


def split(signal: Signal, chuncks_n: int):
    """Yield successive n-sized chunks from lst"""

    for i in range(0, len(signal.wave), chuncks_n):
        yield signal.wave[i : i + chuncks_n]


audio = Signal.load(Path('Task5/Data/test1.wav'))

data = input('Enter:')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print(f'Connected to {HOST}:{PORT}')
    for chunk in split(audio, 1000):
        s.sendall(np.array(chunk, dtype=np.float32).tobytes())
        s.recv(2048).decode()

    s.send(DISCONNECT_MESSAGE.tobytes())

    msg = s.recv(1024)
    print('Result: ', msg.decode())
    s.close()
