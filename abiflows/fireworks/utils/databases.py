# coding: utf-8
"""
Utilities for database insertion
"""
import gridfs
import json
import pymongo
import paramiko
import os
import stat
import shutil

from monty.json import MSONable


class MongoDatabase(MSONable):
    """
    MongoDB database class for access, insertion, update, ... in a MongoDB database
    """

    def __init__(self, host, port, database, username, password, collection, gridfs_collection=None):
        self._host = host
        self._port = port
        self._database = database
        self._username = username
        self._password = password
        self._collection = collection
        self._gridfs_collection = gridfs_collection
        self._connect()

    def _connect(self):
        self.server = pymongo.MongoClient(host=self._host, port=self._port)
        self.database = self.server[self._database]
        if self._username:
            self.database.authenticate(name=self._username, password=self._password)
        self.collection = self.database[self._collection]
        if self._gridfs_collection is not None:
            self.gridfs = gridfs.GridFS(self.database, collection=self._gridfs_collection)
        else:
            self.gridfs = None

    def insert_entry(self, entry, gridfs_msonables=None):
        if gridfs_msonables is not None:
            for entry_value, msonable_object in gridfs_msonables.items():
                dict_str = json.dumps(msonable_object.as_dict())
                file_obj = self.gridfs.put(dict_str, encoding='utf-8')
                entry[entry_value] = file_obj
        self.collection.insert(entry)

    def get_entry(self, criteria):
        count = self.collection.find(criteria).count()
        if count == 0:
            raise ValueError("No entry found with criteria ...")
        elif count > 1:
            raise ValueError("Multiple entries ({:d}) found with criteria ...".format(count))
        return self.collection.find_one(criteria)

    def save_entry(self, entry):
        if '_id' not in entry:
            raise ValueError('Entry should contain "_id" field to be saved')
        self.collection.save(entry)

    def update_entry(self, query, entry_update, gridfs_msonables=None):
        count = self.collection.find(query).count()
        if count != 1:
            raise RuntimeError("Number of entries != 1, found : {:d}".format(count))
        entry = self.collection.find_one(query)
        entry.update(entry_update)
        if gridfs_msonables is not None:
            for entry_value, msonable_object in gridfs_msonables.items():
                if entry_value in entry:
                    backup_current_entry_value = str(entry_value)
                    backup_number = 1
                    while True:
                        if backup_number > 10:
                            raise ValueError('Too many backups (10) for object with entry name "{}"'.format(entry_value))
                        if backup_current_entry_value in entry:
                            backup_current_entry_value = '{}_backup_{:d}'.format(entry_value, backup_number)
                            backup_number += 1
                            continue
                        entry[backup_current_entry_value] = entry[entry_value]
                        break
                dict_str = json.dumps(msonable_object.as_dict())
                file_obj = self.gridfs.put(dict_str, encoding='utf-8')
                entry[entry_value] = file_obj
        self.collection.save(entry)

    def as_dict(self):
        """
        Json-serializable dict representation of a MongoDatabase
        """
        dd = {"@module": self.__class__.__module__,
              "@class": self.__class__.__name__,
              "host": self._host,
              "port": self._port,
              "database": self._database,
              "username": self._username,
              "password": self._password,
              "collection": self._collection,
              "gridfs_collection": self._gridfs_collection}
        return dd

    @classmethod
    def from_dict(cls, d):
        return cls(host=d['host'], port=d['port'], database=d['database'],
                   username=d['username'], password=d['password'], collection=d['collection'],
                   gridfs_collection=d['gridfs_collection'])


class StorageServer(MSONable):
    """
    Storage server class for moving files to/from a given server
    """
    REMOTE_SERVER = 'REMOTE_SERVER'
    LOCAL_SERVER = 'LOCAL_SERVER'

    def __init__(self, hostname, port=22, username=None, password=None, server_type=REMOTE_SERVER):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.server_type = server_type
        # self.connect()

    def connect(self):
        if self.server_type == self.REMOTE_SERVER:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.load_system_host_keys()
            self.ssh_client.connect(hostname=self.hostname, port=self.port,
                                    username=self.username, password=self.password)
            self.sftp_client = self.ssh_client.open_sftp()

    def disconnect(self):
        if self.server_type == self.REMOTE_SERVER:
            self.sftp_client.close()
            self.ssh_client.close()

    def remotepath_exists(self, path):
        try:
            self.sftp_client.stat(path)
        except IOError as e:
            if e[0] == 2:
                return False
            raise
        else:
            return True

    def remote_makedirs(self, path):
        head, tail = os.path.split(path)
        if not tail:
            head, tail = os.path.split(head)
        if head and tail and not self.remotepath_exists(path=head):
            self.remote_makedirs(head)
            if tail == '.':
                return
        self.sftp_client.mkdir(path=path)

    def put(self, localpath, remotepath, overwrite=False, makedirs=True):
        if self.server_type == self.REMOTE_SERVER:
            self.connect()
            if not os.path.exists(localpath):
                raise IOError('Local path "{}" does not exist'.format(localpath))
            if not overwrite and self.remotepath_exists(remotepath):
                raise IOError('Remote path "{}" exists'.format(remotepath))
            rdirname, rfilename = os.path.split(remotepath)
            if not rfilename or rfilename in ['.', '..']:
                raise IOError('Remote path "{}" is not a valid filepath'.format(remotepath))
            if not self.remotepath_exists(rdirname):
                if makedirs:
                    self.remote_makedirs(rdirname)
                else:
                    raise IOError('Directory of remote path "{}" does not exists and '
                                  '"makedirs" is set to False'.format(remotepath))
            sftp_stat = self.sftp_client.put(localpath=localpath, remotepath=remotepath)
            self.disconnect()
            return sftp_stat
        elif self.server_type == self.LOCAL_SERVER:
            if not os.path.exists(localpath):
                raise IOError('Source path "{}" does not exist'.format(localpath))
            if os.path.exists(remotepath) and not overwrite:
                raise IOError('Dest path "{}" exists'.format(remotepath))
            if not os.path.isfile(localpath):
                raise NotImplementedError('Only files can be copied in LOCAL_SERVER mode.')
            shutil.copyfile(src=localpath, dst=remotepath)
        else:
            raise ValueError('Server type "{}" is not allowed'.format(self.server_type))

    def get(self, remotepath, localpath=None, overwrite=False, makedirs=True):
        if self.server_type == self.REMOTE_SERVER:
            self.connect()
            if not self.remotepath_exists(remotepath):
                raise IOError('Remote path "{}" does not exist'.format(remotepath))
            if localpath is None:
                head, tail = os.path.split(remotepath)
                localpath = tail
            localpath = os.path.expanduser(localpath)
            if not overwrite and os.path.exists(localpath):
                raise IOError('Local path "{}" exists'.format(localpath))
            # Check if the remotepath is a regular file (right now, this is the only option that is implemented,
            #  directories should be implemented, symbolic links should be handled in some way).
            remotepath_stat = self.sftp_client.stat(remotepath)
            if stat.S_ISREG(remotepath_stat.st_mode):
                sftp_stat = self.sftp_client.get(remotepath, localpath)
            else:
                raise NotImplementedError('Remote path "{}" is not a regular file'.format(remotepath))
            self.disconnect()
            return sftp_stat
        elif self.server_type == self.LOCAL_SERVER:
            if not os.path.exists(remotepath):
                raise IOError('Source path "{}" does not exist'.format(remotepath))
            if os.path.exists(localpath) and not overwrite:
                raise IOError('Dest path "{}" exists'.format(localpath))
            if not os.path.isfile(remotepath):
                raise NotImplementedError('Only files can be copied in LOCAL_SERVER mode.')
            shutil.copyfile(src=remotepath, dst=localpath)
        else:
            raise ValueError('Server type "{}" is not allowed'.format(self.server_type))

    def as_dict(self):
        """
        Json-serializable dict representation of a StorageServer
        """
        dd = {"@module": self.__class__.__module__,
              "@class": self.__class__.__name__,
              "hostname": self.hostname,
              "port": self.port,
              "username": self.username,
              "password": self.password,
              "server_type": self.server_type}
        return dd

    @classmethod
    def from_dict(cls, d):
        return cls(hostname=d['hostname'], port=d['port'],
                   username=d['username'], password=d['password'],
                   server_type=d['server_type'] if 'server_type' in d else cls.REMOTE_SERVER)
