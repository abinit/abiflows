# coding: utf-8
"""Flows server."""
from __future__ import print_function, division, unicode_literals

import os
import json
import threading

from six.moves.socketserver import BaseRequestHandler, TCPServer, ThreadingMixIn
from monty.collections import AttrDict
from monty.io import FileLock
#from abipy import abilab

# See also
#http://stackoverflow.com/questions/20695241/how-to-add-logging-to-a-file-with-timestamps-to-a-python-tcp-server-for-raspberr?lq=1

import logging
logger = logging.getLogger(__name__)


def daemonize_server(remove_pid=False):
    """
    Args:
        remove_pid: True if the server pid file should be removed before starting new server.
    """
    import daemon
    if remove_pid:
        try:
            os.remove(AbiFlowsServer.pid_path)
        except OSError as exc:
            logger.critical("Could not remove pid_path: %s", exc)

    #with daemon.DaemonContext():
    if True:
        server = AbiFlowsServer()
        server.start()

        #server.write_pid_file()
        #server_thread = threading.Thread(target=server.serve_forever)
        #server_thread.daemon = True
        #print("Server loop running in thread:", server_thread.name)
        #server_thread.start()
        #server_thread.join()


def kill_server():
    d = AbiFlowsServer.read_pid_file()
    if d is None:
        print("Cannot read pid file. Returning")
        return -1

    try:
        os.remove(d.pid_path)
    except OSError as exc:
        logger.critical("Could not remove pid_path: %s", exc)

    # Should try to send SIGINT first!
    return os.system("kill -9 %d" % d.pid)


class AbiFlowsServerError(Exception):
    """Exceptions raised by :class:`AbiFlowsServer`"""


#class AbiFlowsServer(TCPServer):
class AbiFlowsServer(ThreadingMixIn, TCPServer):

    Error = AbiFlowsServerError

    # This one requires root privileges
    #pid_path = os.path.join("/var", "run", "abiflows.pid")
    pid_path = os.path.join(os.getenv("HOME"), ".abinit", "abipy", "abiflows.pid")

    def __init__(self, host="localhost", port=9998, **kwargs):
        self.host, self.port = host, port

        TCPServer.__init__(self, (self.host, self.port), MyTCPHandler)

        self.poll_interval = kwargs.pop("poll_interval", 0.5)

    @classmethod
    def read_pid_file(cls):
        if not os.path.exists(cls.pid_path):
            return None

        with open(cls.pid_path, "rt") as fh:
            d = json.load(fh)
            d["pid_path"] = cls.pid_path

        return AttrDict(**d)

    def write_pid_file(self):
        """
        This function checks if we are already running the AbiPy |Flow| with a :class:`PyFlowScheduler`.
        Raises: Flow.Error if the pid file of the scheduler exists.
        """
        if os.path.exists(self.pid_path):
            raise self.Error("""\n\
                pid_path
                %s
                already exists. There are two possibilities:

                   1) There's an another instance of PyFlowScheduler running
                   2) The previous scheduler didn't exit in a clean way

                To solve case 1:
                   Kill the previous scheduler (use 'kill pid' where pid is the number reported in the file)
                   Then you can restart the new scheduler.

                To solve case 2:
                   Remove the pid_path and restart the scheduler.

                Exiting""" % self.pid_path)

        # Make dir and file if not present.
        if not os.path.exists(os.path.dirname(self.pid_path)):
            os.makedirs(os.path.dirname(self.pid_path))

        import json
        d = dict(
            pid=os.getpid(),
            host=self.host,
            port=self.port,
        )

        #d["flows_db"] = dict(

        #)

        with FileLock(self.pid_path):
            with open(self.pid_path, "wt") as fh:
                json.dump(d, fh)

    def serve_forever(self):
        """
        Handle requests until an explicit shutdown() request.
        Poll for shutdown every poll_interval seconds. Ignores self.timeout.
        """
        TCPServer.serve_forever(self, poll_interval=self.poll_interval)

    def shutdown(self):
        print("Shutdown server")
        TCPServer.shutdown(self)
        #try:
        #    os.remove(self.pid_path)
        #except OSError as exc:
        #    logger.critical("Could not remove pid_path: %s", exc)
        print("Shutdown done")

    def mongo_connect(self):
        """
        Establish a connection with the MongoDb database.
        """
        from pymongo import MongoClient

        #if self.host and self.port:
        #    client = MongoClient(host=config.host, port=config.port)
        #else:
        client = MongoClient()

        self.flows_db = db = client["abiflows"]

        # Authenticate if needed
        #if self.user and self.password:
        #    db.autenticate(self.user, password=self.password)

    def start(self):
        #self.mongo_connect()
        self.write_pid_file()

        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        #server_thread = threading.Thread(target=server.serve_forever)
        server_thread = threading.Thread(target=self.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        print("Server loop running in thread:", server_thread.name)
        server_thread.start()

        server_thread.join()

        # Activate the server
        #self.serve_forever()


class MyTCPHandler(BaseRequestHandler):
    """
    The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        data = self.request.recv(1024)
        print("{} wrote: {}".format(self.client_address[0], data))

        if data == "shutdown":
            return self.server.shutdown()

        # just send back the same data, but upper-cased
        self.request.sendall(data.upper())

        # Reconstruct the `Flow` from pickle.
        #flow = abilab.Flow.pickle_loads(data)
        #flow.allocate(workdir=)

