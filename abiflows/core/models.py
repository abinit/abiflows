# coding: utf-8
"""Object-Document mapper"""
from __future__ import print_function, division, unicode_literals, absolute_import

import os
import six
import collections
import shutil
import gzip

from tempfile import mkstemp, TemporaryFile, NamedTemporaryFile
from monty.json import MontyDecoder
from monty.functools import lazy_property
from mongoengine import *
from mongoengine.fields import GridFSProxy
from mongoengine.base.datastructures import BaseDict
from abipy import abilab

import logging
logger = logging.getLogger(__name__)


class AbiGridFSProxy(GridFSProxy):

    def abiopen(self):
        """Dump the gridfs data to a temporary file and use ``abiopen`` to open the file."""
        _, filepath = mkstemp(suffix='.' + self.abiext, text=self.abiform == "t")

        with open(filepath , "w" + self.abiform) as fh:
            fh.write(self.read())

        return abilab.abiopen(filepath)


class AbiFileField(FileField):
    """
    Extend ``FileField``. Use customized version of proxy_class so that
    we can use ``abiopen`` to construct the AbiPy object from the gridfs content.
    """
    proxy_class = AbiGridFSProxy

    def __init__(self, **kwargs):
        self.abiext = kwargs.pop("abiext")
        self.abiform = kwargs.pop("abiform")

        super(AbiFileField, self).__init__(**kwargs)

    def _monkey_patch_proxy(self, proxy):
        """
        Monkey patch the proxy adding ``abiext`` and ``abiform``.
        so that we know how to open the file with ``abiopen``.
        """
        proxy.abiext, proxy.abiform = self.abiext, self.abiform
        return proxy

    def get_proxy_obj(self, **kwargs):
        proxy = super(AbiFileField, self).get_proxy_obj(**kwargs)
        return self._monkey_patch_proxy(proxy)

    def to_python(self, value):
        if value is not None:
            proxy = super(AbiFileField, self).to_python(value)
            return self._monkey_patch_proxy(proxy)


class GzipGridFSProxy(GridFSProxy):
    """
    Proxy object to handle writing and reading of files to and from GridFS.
    Files are compressed with gzip before being saved in the GridFS.
    To decompress the object exposes an unzip method.
    """

    def put(self, file_obj, **kwargs):
        # try to check if the file is already zipped.
        # in that case just handle as a normal file
        try:
            import magic
            # check that the module is actually python-magic and not libmagic
            f = getattr(magic, "from_buffer", None)
            if f and callable(f):
                if magic.from_buffer(file_obj.read(3), mime=True) == "application/gzip":
                    file_obj.seek(0)
                    return super(GzipGridFSProxy, self).put(file_obj, **kwargs)
                        
            file_obj.seek(0)

        except ImportError:
            logger.info("No python-magic library available. Skipping check...")

        field = self.instance._fields[self.key]
        # Handle nested fields
        if hasattr(field, 'field') and isinstance(field.field, FileField):
            field = field.field

        with TemporaryFile() as tmp_file:
            tmp_gz = gzip.GzipFile(fileobj=tmp_file, compresslevel=field.compresslevel, mode='w+b')
            shutil.copyfileobj(file_obj, tmp_gz)
            tmp_gz.close()
            tmp_file.seek(0)

            return super(GzipGridFSProxy, self).put(tmp_file, **kwargs)

    def write(self, *args, **kwargs):
        raise RuntimeError("Please use \"put\" method instead")

    def writelines(self, *args, **kwargs):
        raise RuntimeError("Please use \"put\" method instead")

    def to_gzip(self):
        return gzip.GzipFile(fileobj=self.get())

    def unzip(self, filepath=None, mode='w+b'):

        with open(filepath, mode) as f:
            f.write(self.to_gzip().read())
            return f


class GzipFileField(FileField):
    """
    A GridFS storage field with automatic compression of the file with gzip.
    """

    proxy_class = GzipGridFSProxy

    def __init__(self, compresslevel=9, **kwargs):
        self.compresslevel = compresslevel
        super(GzipFileField, self).__init__(**kwargs)


class AbiGzipFSProxy(GzipGridFSProxy):
    """
    Proxy object to handle writing and reading of abinit related files to and from GridFS.
    Files are compressed with gzip before being saved in the GridFS.
    To decompress the object exposes an unzip method.
    """

    def abiopen(self):
        """Dump the unzipped gridfs data to a temporary file and use `abiopen` to open the file."""
        field = self.instance._fields[self.key]
        # Handle nested fields
        if hasattr(field, 'field') and isinstance(field.field, FileField):
            field = field.field

        mode = 'w+'
        if field.abiform == 'b':
            mode += 'b'
        with NamedTemporaryFile(suffix='.' + field.abiext, mode=mode) as tmp_abifile:
            tmp_abifile.write(self.to_gzip().read())
            return abilab.abiopen(tmp_abifile.name)


class AbiGzipFileField(GzipFileField):
    """
    A GridFS storage field for abinit related files with automatic compression of the file with gzip.
    """

    proxy_class = AbiGzipFSProxy

    def __init__(self, abiext, abiform, compresslevel=9, **kwargs):
        self.abiext =abiext
        self.abiform = abiform
        super(AbiGzipFileField, self).__init__(compresslevel=compresslevel, **kwargs)


class MongoFiles(EmbeddedDocument):
    """
    Document with the output files produced by the AbiPy |Task|
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
        """Add to GridFs the files produced in the ``outdir`` of the node."""
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
            #print("in out")
            new.output_file.put(node.output_file.read())

        return new

    def delete(self):
        """Delete gridFs files"""
        for field in self._fields.values():
            if not isinstance(field, AbiFileField): continue
            value = getattr(self, field.name)
            if hasattr(value, "delete"):
                print("Deleting %s" % field.name)
                value.delete()


class MSONDict(BaseDict):
    def to_mgobj(self):
        return MontyDecoder().process_decoded(self)


class MSONField(DictField):

    def __get__(self, instance, owner):
        """Descriptor for retrieving a value from a field in a document.
        """
        value = super(MSONField, self).__get__(instance, owner)
        if isinstance(value, BaseDict):
            value.__class__ = MSONDict

        # print("value:", type(value))
        return value

    #def to_python(self, value):
    #    #print("to value:", type(value))
    #    if isinstance(value, collections.Mapping) and "@module" in value:
    #        value = MontyDecoder().process_decoded(value)
    #    else:
    #        value = super(MSONField, self).to_python(value)

    #    print("to value:", type(value))
    #    return value

    #def to_mongo(self, value):
    #    #print(value.as_dict())
    #    return value.as_dict()


class MongoTaskResults(EmbeddedDocument):
    """Document with the most important results produced by the AbiPy |Task|"""

    #meta = {'allow_inheritance': True}

    #: The initial input structure for the calculation in the pymatgen json representation
    #initial_structure = DictField(required=True)
    initial_structure = MSONField(required=True)

    #: The final relaxed structure in a dict format.
    final_structure = DictField(required=True)

    @classmethod
    def from_task(cls, task):
        # TODO Different Documents depending on task.__class__ or duck typing?
        #initial_structure = MSONField().to_mongo(task.input.structure.as_dict())
        initial_structure = task.input.structure.as_dict()
        #print(type(initial_structure))
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

    meta = {
        'allow_inheritance': True,
        "collection": "flowdata",
    }
    #meta = {'meta': True}

    node_id = LongField(required=True)
    node_class = StringField(required=True)
    status = StringField(required=True)
    workdir = StringField(required=True)

    #date_modified = DateTimeField(default=datetime.datetime.now)

    @classmethod
    def from_node(cls, node):
        return cls(
            node_class=node.__class__.__name__,
            node_id=node.node_id,
            status=str(node.status),
            workdir=node.workdir,
        )


class MongoEmbeddedNode(EmbeddedDocument):

    meta = {'allow_inheritance': True}
    #meta = {'meta': True}

    node_id = LongField(required=True)

    node_class = StringField(required=True)

    status = StringField(required=True)

    workdir = StringField(required=True)

    #date_modified = DateTimeField(default=datetime.datetime.now)

    @classmethod
    def from_node(cls, node):
        return cls(
            node_class=node.__class__.__name__,
            node_id=node.node_id,
            status=str(node.status),
            workdir=node.workdir,
        )


class MongoTask(MongoEmbeddedNode):
    """Document associated to an AbiPy |Task|"""

    input = DictField(required=True)
    input_str = StringField(required=True)

    # Abinit events.
    report = DictField(required=True)
    num_warnings = IntField(required=True, help_text="Number of warnings")
    num_errors = IntField(required=True, help_text="Number of errors")
    num_comments =  IntField(required=True, help_text="Number of comments")

    #: Total CPU time taken.
    #cpu_time = FloatField(required=True)
    #: Total wall time taken.
    #wall_time = FloatField(required=True)

    results = EmbeddedDocumentField(MongoTaskResults)

    outfiles = EmbeddedDocumentField(MongoFiles)

    #@property
    #def num_warnings(self):
    #    """Number of warnings reported."""
    #    return self.input.num_warnings

    #@property
    #def num_errors(self):
    #    """Number of errors reported."""
    #    return self.input.num_error

    #@property
    #def num_comments(self):
    #    """Number of comments reported."""
    #    return len(self.comments)

    #@property
    #def is_paw(self):
    #    print("in is paw")
    #    return True

    @classmethod
    def from_task(cls, task):
        """Build the document from an AbiPy |Task| object."""
        new = cls.from_node(task)

        new.input = task.input.as_dict()
        new.input_str = str(task.input)

        # TODO: Handle None!
        report = task.get_event_report()
        for a in ("num_errors", "num_comments", "num_warnings"):
            setattr(new, a, getattr(report, a))
        new.report = report.as_dict()

        new.results = MongoTaskResults.from_task(task)
        new.outfiles = MongoFiles.from_node(task)

        return new


class MongoWork(MongoEmbeddedNode):
    """Document associated to an AbiPy |Work|"""

    #: List of tasks.
    tasks = ListField(EmbeddedDocumentField(MongoTask), required=True)

    #: Output files produced by the work.
    outfiles = EmbeddedDocumentField(MongoFiles)

    @classmethod
    def from_work(cls, work):
        """Build and return the document from a AbiPy |Work| instance."""
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
    """
    Document associated to an AbiPy |Flow|

    Assumptions:
        All the tasks must have the same list of pseudos,
        same chemical formula.
    """

    #: List of works
    works = ListField(EmbeddedDocumentField(MongoWork), required=True)

    #: Output files produced by the flow.
    outfiles = EmbeddedDocumentField(MongoFiles)

    #meta = {
    #    "collection": "flowdata",
    #    #"indexes": ["status", "priority", "created"],
    #}

    @classmethod
    def from_flow(cls, flow):
        """Build and return the document from a AbiPy |Flow| instance."""
        new = cls.from_node(flow)
        new.works = [MongoWork.from_work(work) for work in flow]
        new.outfiles = MongoFiles.from_node(flow)
        #new.assimilated = flow.mongo_assimilate()
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

        Return: AbiPy |Flow| object.
        """
        flow = abilab.Flow.pickle_load(self.workdir)
        #flow.set_mongo_id(self.id)
        return flow

    def delete(self):
        # Remove GridFs files.
        for work in self.works:
            work.outfiles.delete()
            #work.delete()
            for task in work:
                #task.delete()
                task.outfiles.delete()

        self.delete()

    @queryset_manager
    def completed(doc_cls, queryset):
        return queryset.filter(status="Completed")

    #@queryset_manager
    #def running(doc_cls, queryset):
    #    return queryset.filter(status__in=["AbiCritical", "QCritical", "Error",])

    #@queryset_manager
    #def paw_flows(doc_cls, queryset):
    #    return queryset.filter(is_paw=True)

    #@queryset_manager
    #def nc_flows(doc_cls, queryset):
    #    return queryset.filter(is_nc=True)
