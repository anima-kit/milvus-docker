### tests/test_integration.py
## Defines integration tests for making sure methods in ./pyfiles/milvus_utils.py can be properly used with Milvus server

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Imports
# Third-party modules
import pytest
import unittest
from pymilvus import MilvusClient # type: ignore
from pymilvus.exceptions import MilvusException # type: ignore
from pymilvus.client.search_result import SearchResult  # type: ignore

# Internal modules
from pyfiles.milvus_utils import (
    MilvusClientInit, 
    uri, 
    data_ex, 
    query_list, 
    lim_results
)

## Define collection name to use purely for tests
collection_name = '_test_collection'


class TestMilvusClientIntegration(unittest.TestCase):
    """
    Integration tests for MilvusClientInit against a real Milvus server.

    These tests ensure that the methods in `pyfiles.milvus_utils` work correctly when communicating with a running instance of the Milvus server. 
    They verify basic functionality such as:
    
    - Client initialization
    - Listing collections
    - Creating and deleting collections
    - Inserting data
    - Performing full text search
    
    All tests require a running Milvus server at http://localhost:19530.
    """

    ## Class resources
    # This sets up resources that are used for each test in the class
    @classmethod
    def setUpClass(cls):
        """
        Set up class-level fixtures once for all tests.

        This setup ensures that common values are defined before any individual test runs.

        Variables
        ------------
            uri: str
                URI of the Milvus server.
                Defaults to 'http://localhost:19530'.
        """
        cls.test_uri = uri


    ## Set up for each test
    # This sets up whatever needs to be run before each test
    # In our case, we want to make sure the Milvus server is available, and if it isn't we skip the current test and move onto the next
    def setUp(self):
        """
        Set up test fixtures before each test method.

        Verifications
        ------------
            Milvus server is accessible by attempting a `list_collections` request.
            If the server is unreachable, this test will be skipped with a message.

        Raises
        ------------
            unittest.SkipTest
                When the Milvus server is not reachable at http://localhost:19530.
        """
        error_message = "Milvus server not accessible at http://localhost:19530."
        # Verify Milvus server is accessible before running tests
        try:
            client = MilvusClient(uri=uri)
            collections = client.list_collections()
            # If collections isn't a list, print error message
            self.assertTrue(isinstance(collections, list), error_message)
        # If we get connection error, skip the test
        except MilvusException:
            self.skipTest(error_message)


    ## Test initializing client
    # Can run this test 1st because client will need to be initialized for every other test
    @pytest.mark.order(1)
    def test_client_initialization(self):
        """
        Test that MilvusClientInit can be initialized successfully.

        This ensures that the `MilvusClientInit` class can be instantiated correctly with a given URI.

        Verifications
        ------------
            MilvusClientInit object is not None.
            URI matches the expected value.
            Internal MilvusClient has been created.
        
        Raises
        ------------
            Exception: 
                If any verifications fail.
        """
        try:
            client = MilvusClientInit(uri=self.test_uri)
            self.assertIsNotNone(client)
            self.assertEqual(client.uri, self.test_uri)
            self.assertIsNotNone(client.client)
        except Exception as e:
            self.fail(f"Failed to initialize MilvusClientInit: {e}")


    @pytest.mark.order(2)
    ## Test listing collections
    def test_list_collections(self):
        """
        Test listing Milvus client collections.

        Ensures that the `list_collections` method returns a list.

        Verifications
        ------------
            Return type is a list.
        
        Raises
        ------------
            Exception: 
                If any verifications fail.
        """
        # We already know whether client can be properly initialized with first test, so we can leave it out of try - except blocks
        client = MilvusClientInit(uri=self.test_uri)
        try:
            collections = client.list_collections()
            # Should be a list of models names
            self.assertIsInstance(collections, list)
        except Exception as e:
            self.fail(f"Failed to list collections: {e}")


    @pytest.mark.order(3)
    ## Test creating collection
    def test_create_collection(self):
        """
        Test creating collection for Milvus client.

        Ensures that the `create_collection` method properly creates a collection.

        Verifications
        ------------
            Added collection is in collection list.
        
        Raises
        ------------
            Exception: 
                If any verifications fail.
        """
        client = MilvusClientInit(uri=self.test_uri)
        try:
            client.create_collection(name=collection_name)
            collections = client.list_collections()
            # collection_name should be in collections
            self.assertIn(collection_name, collections)
        except Exception as e:
            self.fail(f"Failed to create collection: {e}")


    @pytest.mark.order(4)
    ## Test inserting data
    def test_insert(self):
        """
        Test inserting data into collection.

        Ensures that the `insert` method returns a dictionary.

        Verifications
        ------------
            Return type is a dictionary.
        
        Raises
        ------------
            Exception: 
                If any verifications fail.
        """
        client = MilvusClientInit(uri=self.test_uri)
        try:
            results = client.insert(name=collection_name, data=data_ex)
            # Results should be a dictionary
            self.assertIsInstance(results, dict)
        except Exception as e:
            self.fail(f"Failed to insert data: {e}")    


    @pytest.mark.order(5)
    ## Test performing full text search
    def test_full_text_search(self):
        """
        Test performing full text search on inserted data.

        Ensures that the `full_text_search` method returns a `SearchResult`.

        Verifications
        ------------
            Return type is a `SearchResult`.
        
        Raises
        ------------
            Exception: 
                If any verifications fail.
        """
        client = MilvusClientInit(uri=self.test_uri)
        try:
            results = client.full_text_search(
                name=collection_name, 
                query_list=query_list,
                limit=lim_results
            )
            # Results should be a `SearchResult`
            self.assertIsInstance(results, SearchResult)
        except Exception as e:
            self.fail(f"Failed to perform a full text search: {e}")  


    @pytest.mark.order(6)
    ## Test dropping collection
    def test_drop_collection(self):
        """
        Test dropping collection for Milvus client.

        Ensures that the `drop_collection` method properly drops a collection.

        Verifications
        ------------
            Removed collection is not in collection list.
        
        Raises
        ------------
            Exception: 
                If any verifications fail.
        """
        client = MilvusClientInit(uri=self.test_uri)
        try:
            client.drop_collection(name=collection_name)
            collections = client.list_collections()
            # collection_name shouldn't be in collections
            self.assertNotIn(collection_name, collections)
        except Exception as e:
            self.fail(f"Failed to drop collection: {e}")


    ## Tear down after finishing each test
    def tearDown(self):
        """
        Clean up after each test method.

        Currently performs no actions but serves as a placeholder for future cleanup logic.
        """
        # Nothing special to clean up for this test
        pass


    ## Tear down after finishing all tests
    @classmethod
    def tearDownClass(cls):
        """
        Clean up class-level fixtures.

        Currently performs no actions but serves as a placeholder for future cleanup logic.
        """
        # Nothing special to clean up for this test
        pass