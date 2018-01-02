from __future__ import print_function, division, unicode_literals, absolute_import

import os
import unittest

from abiflows.core.testing import AbiflowsTest, has_mongodb, TESTDB_NAME
from abiflows.database.mongoengine.utils import *
from mongoengine import Document, StringField
from mongoengine.context_managers import switch_db
from pymongo import MongoClient


class TestDatabaseData(AbiflowsTest):

    def setUp(self):
        self.setup_mongoengine()

    @classmethod
    def tearDownClass(cls):
        cls.teardown_mongoengine()

    def test_class(self):
        db_data = DatabaseData(TESTDB_NAME,host="localhost", port=27017, collection="test_collection", username="user",
                               password="password")

        self.assertMSONable(db_data)
        d = db_data.as_dict_no_credentials()
        assert "username" not in d
        assert "password" not in d

    @unittest.skipUnless(has_mongodb(), "A local mongodb is required.")
    def test_connection(self):
        db_data = DatabaseData(TESTDB_NAME, collection="test_collection")

        db_data.connect_mongoengine()

        class TestDocument(Document):
            test = StringField()

        with db_data.switch_collection(TestDocument) as TestDocument:
            TestDocument(test="abc").save()

        # check the collection with pymongo
        client = MongoClient()
        db = client[TESTDB_NAME]
        collection = db[db_data.collection]
        documents = collection.find()
        assert documents.count() == 1
        assert documents[0]['test'] == "abc"

    @unittest.skipUnless(has_mongodb(), "A local mongodb is required.")
    def test_connection_alias(self):

        test_db_name_2 = TESTDB_NAME+"_2"
        try:
            db_data2 = DatabaseData(test_db_name_2, collection="test_collection")
            db_data2.connect_mongoengine(alias="test_alias")

            db_data1 = DatabaseData(TESTDB_NAME, collection="test_collection")
            db_data1.connect_mongoengine()

            class TestDocument(Document):
                test = StringField()

            with switch_db(TestDocument, "test_alias") as TestDocument:
                with db_data2.switch_collection(TestDocument) as TestDocument:
                    TestDocument(test="abc").save()

            # check the collection with pymongo
            client = MongoClient()
            db = client[test_db_name_2]
            collection = db[db_data2.collection]
            documents = collection.find()
            assert documents.count() == 1
            assert documents[0]['test'] == "abc"
        finally:
            if self._connection:
                self._connection.drop_database(test_db_name_2)
