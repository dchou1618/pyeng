#!/usr/bin/env python3

# echo client

import socket, rsaKey, sys
from rsaKey import *

def mainClient(key):
    HOST,PORT = "127.0.0.1",65432 # localhost, port to listen on
    # connects to initialized host, port and sends the key value, printing to the console the received data
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST,PORT))
        s.sendall(str.encode(key))
        data = s.recv(1024)
        print("Received: {}".format(data))
