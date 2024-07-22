import multiprocessing
import threading
import socket
import time
import vosk
from pathlib import Path
import recogniser
import numpy as np


def pretty_address(address: tuple[str, str]):
    return f'{address[0]}:{address[1]}'


class Server:
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
            funcs.update({func.__name__: func})
            return func
        else:
            return funcs

    def __init__(self, socket: socket.socket) -> None:
        self.socket = socket
        self.is_stopped = False
        self._address_pull: dict[str, multiprocessing.Process] = {}

    def accept_client(self, stop_func: 'function'):
        """
        Loop for accept new clients in separate thread, start new processes

        :param stop_func: Event for stop thread
        """
        while not stop_func():
            try:
                client_socket, address = self.socket.accept()
            except:
                break
            print(f'Connected: {pretty_address(address)}')

            process = multiprocessing.Process(
                target=receive,
                args=(client_socket, address),
                # daemon=True,
            )
            process.start()
            self._address_pull[pretty_address(address)] = process

    def start(self):
        """Start server"""

        self.socket.bind((HOST, PORT))
        self.socket.listen()

        print("Start server")
        self._accept_thread = threading.Thread(
            target=self.accept_client,
            args=(lambda: self.is_stopped,),
            daemon=True,
        )
        self._accept_thread.start()

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

    def del_client(self, address: tuple[str, str]):
        del self._address_pull[pretty_address(address)]

    def execute(self, command: str):
        parsed_command = command.strip().replace(' ', '_')
        commands = self._command(get=True)
        if parsed_command in commands.keys():
            return commands.get(parsed_command)(self)
        else:
            print('Unknown command')

    @_command
    def help(self):
        """Help"""

        command_dict = self._command(get=True)
        for command_name, command in command_dict.items():
            print(f"{command_name} -- {command.__doc__}")

    @_command
    def clients_list(self):
        """List of active clients"""

        self._address_pull = {
            key: process for key, process in self._address_pull if process.is_alive()
        }

        print('\n'.join(self._address_pull.keys()).strip())


def receive(connection: socket.socket, address: tuple[str, str]):
    """
    Working function connect and recognize

    :param connection: Client socket
    :param address: Client address
    """
    res = []
    with connection:
        print(f'Start recognize for {pretty_address(address)}')
        while True:
            data = connection.recv(2048)
            data = np.frombuffer(data, dtype=np.float32)
            if np.average(data) > 1:
                break
            res.append(data)
            connection.sendall('Chunk received'.encode())
        time.sleep(1)

        print(f'End recognize for {pretty_address(address)}')
        connection.sendall(str(res).encode())


DISCONNECT_MESSAGE = np.array([999], dtype=np.float32)
HOST = "127.0.0.1"
PORT = 65432

if __name__ == '__main__':
    # MODEL = vosk.Model(Path(r"D:\Works\TechTasks\vosk-model-ru-0.42").as_posix())
    # MODEL_SAMPLE_RATE = 8000

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server = Server(server_socket)
        server.start()

        while True:
            command = input()
            callback = server.execute(command)
            if callback == -1:
                break
