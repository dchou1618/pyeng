#!/usr/bin/env python3

# echo server

import socket, selectors, sys

def mainServer():
    HOST, PORT = '127.0.0.1', 65432 # localhost & port to listen on
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    conn.sendall(data)
    except Exception as e:
        print("Exception {} occured".format(e))
if __name__ == "__main__":
    mainServer()
