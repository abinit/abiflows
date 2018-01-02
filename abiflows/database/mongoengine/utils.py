# coding: utf-8
"""
Utilities to handle mongoengine classes and connections.
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import contextlib

from pymatgen.util.serialization import pmg_serialize
from monty.json import MSONable
from mongoengine import connect
from mongoengine.context_managers import switch_collection
from mongoengine.connection import DEFAULT_CONNECTION_NAME


class DatabaseData(MSONable):
    """
    Basic class providing data to connect to a collection in the database and switching to that collection.
    Wraps mongoengine's connect and switch_collection functions.
    """

    def __init__(self, database, host=None, port=None, collection=None, username=None, password=None):
        """
        Args:
             database: name of the database
             host: the host name of the mongod instance to connect to
             port: the port that the mongod instance is running on
             collection: name of the collection
             username: username to authenticate with
             password: password to authenticate with
        """
        #TODO handle multiple collections?
        # note: if making collection a list (or a dict), make it safe for mutable default arguments, otherwise there
        # will probably be problems with the switch_collection
        self.database = database
        self.host = host
        self.port = port
        self.collection = collection
        self.username = username
        self.password = password

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d.pop("@module", None)
        d.pop("@class", None)
        return cls(**d)

    @pmg_serialize
    def as_dict(self):
        return dict(database=self.database, host=self.host, port=self.port, collection=self.collection,
                    username=self.username, password=self.password)

    @pmg_serialize
    def as_dict_no_credentials(self):
        return dict(database=self.database, host=self.host, port=self.port, collection=self.collection)

    def connect_mongoengine(self, alias=DEFAULT_CONNECTION_NAME):
        """
        Open the connection to the selected database
        """

        return connect(db=self.database, host=self.host, port=self.port, username=self.username,
                       password=self.password, alias=alias)

    @contextlib.contextmanager
    def switch_collection(self, cls):
        """
        Switches to the chosen collection using Mongoengine's switch_collection.
        """

        if self.collection:
            with switch_collection(cls, self.collection) as new_cls:
                yield new_cls
        else:
            yield cls
