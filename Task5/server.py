import multiprocessing
import threading
import socket
import struct

import vosk
from pathlib import Path


import recognizer as recognizer
import numpy as np


def get_pretty_address(address: tuple[str, str]):
    """Get adress pretty string"""

    return f'{address[0]}:{address[1]}'


END_MESSAGE = 'END'
RECIEVE_MESSAGE = '_CHUNK_RECIEVED_'


def send_msg(socket: socket.socket, message: bytes):
    """Prefix each message with a 4-byte length (network byte order)"""

    message = struct.pack('>I', len(message)) + message
    socket.sendall(message)


def _recvall(socket: socket.socket, n: int) -> bytearray:
    """Helper function to recv n bytes or return None if EOF is hit"""

    data = bytearray()
    while len(data) < n:
        packet = socket.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def recv_msg(socket: socket.socket) -> bytearray:
    """Read message length and unpack it into an integer"""

    raw_msg_len = _recvall(socket, 4)
    if not raw_msg_len:
        return None
    msglen = struct.unpack('>I', raw_msg_len)[0]
    return _recvall(socket, msglen)


class Server:

    def __init__(self, socket: socket.socket, model) -> None:
        self.model = model
        self.socket = socket
        self.is_stopped = False
        self._address_pull: dict[str, multiprocessing.Process] = {}

    def accept(self, stop_func: 'function'):
        """
        Loop for accept new clients in separate thread, start new processes

        :param stop_func: Event for stop thread
        """
        while not stop_func():
            try:
                client_socket, address = self.socket.accept()
            except:
                break
            print(f'Connected: {get_pretty_address(address)}')

            process = multiprocessing.Process(
                target=connect_client,
                args=(client_socket, address, self.model),
                daemon=True,
            )
            process.start()
            self._address_pull[get_pretty_address(address)] = process
        print('Listening stopped')

    def start(self):
        """Start server"""

        self.socket.bind((HOST, PORT))
        self.socket.listen()

        print("Start server")
        self._accept_thread = threading.Thread(
            target=self.accept,
            args=(lambda: self.is_stopped,),
            daemon=True,
        )
        self._accept_thread.start()

    def del_client(self, address: tuple[str, str]):
        del self._address_pull[get_pretty_address(address)]

    @staticmethod
    def _command(func=None, get=False, funcs: dict[str, 'function'] = {}):
        """
        Wrapper to mark function as commands in server CLI

        :param func: _description_, defaults to None
        :param get: Working mode, get on wrap, defaults to False
        :param funcs: Functions dict, it's enclosure, defaults to {}
        :return: Wrapped function
        """
        if not get:
            funcs.update({func.__name__.replace('_', ' '): func})
            return func
        else:
            return funcs

    def execute_command(self, command: str):
        parsed_command = command.strip()
        commands = self._command(get=True)
        if parsed_command in commands.keys():
            return commands.get(parsed_command)(self)
        else:
            print('Unknown command')

    @_command
    def stop(self):
        """Stop server"""

        print('Stopping server...')
        self.is_stopped = True
        self.socket.close()
        self.socket.detach()
        self._accept_thread.join()
        return -1

    @_command
    def kill(self):
        """Kill all processes"""

        for process in self._address_pull.values():
            process.kill()
        self.stop()
        return -1

    @_command
    def help(self):
        """Help"""

        command_dict = self._command(get=True)
        for command_name, command in command_dict.items():
            print(f"{command_name} -- {command.__doc__}")

    @_command
    def client_list(self):
        """List of active clients"""

        self._address_pull = {
            key: process for key, process in self._address_pull.items() if process.is_alive()
        }

        print('\n'.join(self._address_pull.keys()).strip())


def connect_client(
    client_socket: socket.socket,
    address: tuple[str, str],
    model,
):
    """
    Working function connect and recognize

    :param connection: Client socket
    :param address: Client address
    """
    recieved_audio = []
    with client_socket:
        print(f'Open connection : {get_pretty_address(address)}')

        sample_rate = int(recv_msg(client_socket).decode())

        while True:
            msg = recv_msg(client_socket)

            if not msg or (msg == END_MESSAGE.encode()):
                break
            data = np.frombuffer(msg, dtype=np.float32)
            recieved_audio.append(data)

        wave = np.concatenate(recieved_audio, dtype=np.float32)

        signal = recognizer.Signal(wave=wave, sample_rate=sample_rate)
        # signal.to_wav(Path('Server_save'), 'Ser1')
        # print('Signal saved')
        # text = rec.recognize_signal(signal)
        print(type(model))
        # print(text)

        print(f'Close connection: {get_pretty_address(address)}')
        send_msg(client_socket, 'Disconnect'.encode())


HOST = "127.0.0.1"
PORT = 65432

if __name__ == '__main__':
    MODEL = vosk.Model(Path(r"D:\WORKS\TechTasks\vosk-model-ru-0.42").as_posix())
    MODEL_SAMPLE_RATE = 8000
    rec = recognizer.Recognizer(MODEL, MODEL_SAMPLE_RATE)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server = Server(server_socket, rec._rec)
        server.start()

        while True:
            command = input()
            callback = server.execute_command(command)
            if callback == -1:
                break
