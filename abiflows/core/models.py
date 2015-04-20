# coding: utf-8
"""Object-Document mapper"""
from __future__ import print_function, division, unicode_literals

import os
import six
from abipy import abilab

from mongoengine import *
from mongoengine.fields import GridFSProxy


import logging
logger = logging.getLogger(__name__)


class AbiGridFSProxy(GridFSProxy):

    def abiopen(self):
        """Dump the gridfs data to a temporary file and use `abiopen` to open the file."""
        from tempfile import mkstemp
        _, filepath = mkstemp(suffix='.' + self.abiext, text=True if self.abiext == "t" else False)

        with open(filepath , "w" + self.abiform) as fh:
            fh.write(self.read())

        return abilab.abiopen(filepath)


class AbiFileField(FileField):
    """
    Extend `FileField`. Use customized version of proxy_class so that
    we can use `abiopen` to construct the AbiPy object from the gridfs content.
    """
    proxy_class = AbiGridFSProxy

    def __init__(self, **kwargs):
        self.abiext = kwargs.pop("abiext")
        self.abiform = kwargs.pop("abiform")

        super(AbiFileField, self).__init__(**kwargs)

    def get_proxy_obj(self, **kwargs):
        """
        Monkey path the proxy adding `abiext` and `abiform`.
        so that we know how to open the file in `abiopen`.
        """
        proxy = super(AbiFileField, self).get_proxy_obj(**kwargs)
        proxy.abiext = self.abiext
        proxy.abiform = self.abiform
        return proxy


class MongoFiles(EmbeddedDocument):
    """
    Document with the output files produced by the :class:`Task` 
    (references to GridFs files)
    """
    gsr = AbiFileField(abiext="GSR.nc", abiform="b")
    hist = AbiFileField(abiext="HIST", abiform="b")
    phbst = AbiFileField(abiext="PHBST.nc", abiform="b")
    phdos = AbiFileField(abiext="PHDOS.nc", abiform="b")
    sigres = AbiFileField(abiext="SIGRES.nc", abiform="b")
    mdf = AbiFileField(abiext="MDF.nc", abiform="b")

    ddb = AbiFileField(abiext="DDB", abiform="t")
    output_file = AbiFileField(abiext="abo", abiform="t")

    @classmethod
    def from_node(cls, node):
        """Add to GridFs the files produced in the `outdir` of the node."""
        new = cls()

        for key, field in cls._fields.items():
            if not isinstance(field, AbiFileField): continue
            ext, form = field.abiext, field.abiform

            path = node.outdir.has_abiext(ext)
            if path:
                with open(path, "r" + form) as f:
                    # Example: new.gsr.put(f)
                    fname = ext.replace(".nc", "").lower()
                    proxy = getattr(new, fname)
                    proxy.put(f)
        
        # Special treatment of the main output 
        # (the file is not located in node.outdir)
        if hasattr(node, "output_file"):
            print("out")
            new.output_file.put(node.output_file.read())

        return new


class MongoTaskResults(EmbeddedDocument):
    """Document with the most important results produced by the :class:`Task`"""

    meta = {'allow_inheritance': True}

    #: The initial input structure for the calculation in the pymatgen json representation
    initial_structure = DictField(required=True)

    #: The final relaxed structure in a dict format. 
    final_structure = DictField(required=True)

    @classmethod
    def from_task(cls, task):
        # Differernt Documents depending on task.__class__ or duck typing?
        initial_structure = task.input.structure.as_dict()
        final_structure = initial_structure

        if hasattr(task, "open_gsr"):
            with task.open_gsr() as gsr:
                final_structure = gsr.structure.as_dict()

        new = cls(
            initial_structure=initial_structure,
            final_structure=final_structure,
        )
        return new


class MongoNode(Document):

    meta = {'allow_inheritance': True}
    #meta = {'meta': True}

    node_id = LongField(required=True)

    node_class = StringField(required=True)

    status = IntField(required=True)

    workdir = StringField(required=True)

    #date_modified = DateTimeField(default=datetime.datetime.now)

    @classmethod
    def from_node(cls, node):
        return cls(
            node_class=node.__class__.__name__,
            node_id=node.node_id,
            status=node.status,
            workdir=node.workdir,
        )


class MongoEmbeddedNode(EmbeddedDocument):

    meta = {'allow_inheritance': True}
    #meta = {'meta': True}

    node_id = LongField(required=True)

    node_class = StringField(required=True)

    status = IntField(required=True)

    workdir = StringField(required=True)

    #date_modified = DateTimeField(default=datetime.datetime.now)

    @classmethod
    def from_node(cls, node):
        return cls(
            node_class=node.__class__.__name__,
            node_id=node.node_id,
            status=node.status,
            workdir=node.workdir,
        )


class MongoTask(MongoEmbeddedNode):
    """Document associated to a :class:`Task`"""

    input = DictField(required=True)
    input_str = StringField(required=True)
    #output_str = StringField(required=True)

    event_report = DictField(required=True)
    #corrections =

    results = EmbeddedDocumentField(MongoTaskResults)

    outfiles = EmbeddedDocumentField(MongoFiles)

    @classmethod
    def from_task(cls, task):
        """Build the document from a :class:`Task` instance."""
        new = cls.from_node(task)

        new.input = task.input.as_dict()
        new.input_str = str(task.input)

        # TODO: Handle None!
        report = task.get_event_report()
        new.event_report = report.as_dict()

        new.results = MongoTaskResults.from_task(task)
        new.outfiles = MongoFiles.from_node(task)

        return new


class MongoWork(MongoEmbeddedNode):
    """Document associated to a :class:`Work`"""
    
    #: List of tasks.
    tasks = ListField(EmbeddedDocumentField(MongoTask), required=True)
    #: Output files produced by the work.
    outfiles = EmbeddedDocumentField(MongoFiles)

    @classmethod
    def from_work(cls, work):
        """Build and return the document from a :class:`Work` instance."""
        new = cls.from_node(work)
        new.tasks = [MongoTask.from_task(task) for task in work]
        new.outfiles = MongoFiles.from_node(work)
        return new

    def __getitem__(self, name):
        try:
            # Dictionary-style field of super
            return super(MongoWork, self).__getitem__(name)
        except KeyError:
            # Assume int or slice
            try:
                return self.tasks[name]
            except IndexError:
                raise


class MongoFlow(MongoNode):
    """Document associated to a :class:`Flow`"""

    #: List of works
    works = ListField(EmbeddedDocumentField(MongoWork), required=True)
    #: Output files produced by the flow.
    outfiles = EmbeddedDocumentField(MongoFiles)

    @classmethod
    def from_flow(cls, flow):
        """Build and return the document from a :class:`Flow` instance."""
        new = cls.from_node(flow)
        new.works = [MongoWork.from_work(work) for work in flow]
        new.outfiles = MongoFiles.from_node(flow)
        return new

    def __getitem__(self, name):
        try:
            # Dictionary-style field of super
            return super(MongoFlow, self).__getitem__(name)
        except KeyError:
            # Assume int or slice
            try:
                return self.works[name]
            except IndexError:
                raise

    def pickle_load(self):
        """
        Load the pickle file from the working directory of the flow.

        Return:
            :class:`Flow` instance.
        """
        flow = abilab.Flow.pickle_load(self.workdir)
        #flow.set_mongo_id(self.id)
        return flow 
