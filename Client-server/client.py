import socket


HOST = "127.0.0.1"
PORT = 65432
DISCONNECT_MESSAGE = '!DISCONNECT'
SEPARATOR = '<SEPARATE>'

s = socket.socket()


data = input('Enter:')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print(f'Connected to {HOST}:{PORT}')
    for i in data:
        s.send((i + SEPARATOR).encode())
        print(s.recv(1000).decode())

    s.send(DISCONNECT_MESSAGE.encode())

    msg = s.recv(1024)
    print('Result: ', msg.decode())
    s.close()
