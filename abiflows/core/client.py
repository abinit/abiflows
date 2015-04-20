# coding: utf-8
"""AbiFlows client."""
from __future__ import print_function, division, unicode_literals

import socket

from .server import AbiFlowsServer

class AbiFlowsClientError(Exception):
    """Exceptions raised by :class:`AbiFlowsClient`"""


class AbiFlowsClient(object):

    Error = AbiFlowsClientError

    def __init__(self):
        c = AbiFlowsServer.read_pid_file()
        if c is None:
            raise self.Error("Cannot read pid file %s.\n"
                             "Perhaps the server is not running" % AbiFlowsServer.pid_path)

        print(c)
        self.host, self.port = c.host, c.port

        # Create a socket (SOCK_STREAM means a TCP socket)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()

    def __enter__(self):
        """Support for "with" context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for "with" context."""
        self.close()

    def connect(self):
        """Connect to server"""
        self.sock.connect((self.host, self.port))

    def close(self):
        """shut down"""
        self.sock.close()

    def shutdown_server(self):
        self.sock.sendall("shutdown")

    def send_flow(self, flow):
        #data = flow.pickle_dumps()
        data = flow
        #try:
        # Connect to server and send data
        self.sock.sendall(data + "\n")

        # Receive data from the server and shut down
        received = self.sock.recv(1024)

        print("Sent:     {}".format(data))
        print("Received: {}".format(received))

        return received

