### tests/test_unit_milvus.py
## Defines unit tests for methods in ./pyfiles/milvus_utils.py to be used without a Milvus server

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Imports
# Third-party modules
from unittest import TestCase
from unittest.mock import (
    call, 
    patch, 
    MagicMock
)
from pymilvus import (
    MilvusClient, 
    CollectionSchema
)
from pymilvus.exceptions import MilvusException
from pymilvus.milvus_client.index import IndexParams
from pymilvus.client.search_result import (
    SearchResult, 
    HybridHits, 
    Hit
)

# Internal modules
from pyfiles.milvus_utils import (
    MilvusClientInit, 
    uri, 
    field_params_list, 
    index_params_list, 
    func_bm25, 
    data_ex, 
    query_list, 
    lim_results
)

## Define collection name to use purely for tests
collection_name = '_test_collection'

## Now let's test everything
class TestMilvusClientUnit(TestCase):
    """
    Unit tests for `MilvusClient` class.
    
    This test suite contains unit tests for the `MilvusClient` class of `pyfiles.milvus_utils`, covering:

        - Client initialization
        - Listing collections
        - Creating and deleting collections
        - Inserting data
        - Performing full text search

    All tests use mocking to isolate the class under test from external dependencies.
    """

    def _loop_through_params(
        self, 
        param_name, 
        param_list, 
        method_name,
        method_args, 
        client
    ):
        """
        Loop through possible argument parameters to test error handling for subtests of various methods.
        
        Args
        ------------
            param_name: str
                The name of the parameter to test.
            param_list: list
                A list of invalid parameters to test.
            method_name: str
                The name of the method to test.
                Can be any method within MilvusClientInit.
            method_args: dict
                Dictionary of arguments for the method.
            client: MilvusClientInit
                Client for which methods will be tested.
        
        Asserts
        ------------
            Exception is raised when the method is passed bad argument types.
        """
        for param, description in param_list:
            with self.subTest(param_type=description):
                with self.assertRaises(Exception):
                    method = getattr(client, method_name)
                    method_args[param_name] = param 
                    method(**method_args)


    ## Test successful client initialization
    @patch('pyfiles.milvus_utils.MilvusClient')
    def test_init_success(
        self, 
        mock_client
    ):
        """
        Test successful initialization of MilvusClient with a custom URI.
        
        Verifications
        ------------
            The client initialization method (`_init_client`) is called once.
            The `MilvusClient` is initialized with the correct URI.
        
        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
        
        Asserts
        ------------
            `MilvusClientInit` URI matches the provided URI.
            `MilvusClient` is called exactly once with correct host parameter.
        """
        ## Arrange
        # MilvusClient | Create a mock instance of the client
        mock_client_instance = MagicMock(spec=MilvusClient)
        mock_client.return_value = mock_client_instance

        ## Act
        # MilvusClient call replaced with mock_client
        client = MilvusClientInit(uri=uri)

        ## Assert
        self.assertEqual(client.uri, uri)
        mock_client.assert_called_once_with(uri=uri)


    ## Test unsuccessful client initialization
    @patch('pyfiles.milvus_utils.MilvusClient')
    def test_init_unavailable(
        self, 
        mock_client
    ):
        """
        Test error handling of `MilvusClientInit` when Milvus server unavailable.
        
        Verifications
        ------------
            Initializing the `MilvusClientInit` raises an exception when the Milvus server is unavailable.
            Exception is propagated correctly.
        
        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock error.
        
        Asserts
        ------------
            `MivlusException` is raised when Milvus server is unavailable.
        """
        ## Arrange
        # MilvusClient | Create error
        error = MilvusException
        mock_client.side_effect = error

        ## Act and Assert
        with self.assertRaises(error):
            MilvusClientInit(uri=uri)


    ## Test error handling bad uri
    def test_init_bad_uri(self):
        """
        Test error handling of `MilvusClientInit` when `MilvusClient` is passed a wrong uri type.
        
        Verifications
        ------------
            Initializing the `MilvusClientInit` raises an exception when the `MilvusClient` is passed a wrong URI type.
            Exception is propagated correctly.
        
        Asserts
        ------------
            Exception is raised when `MilvusClientInit` is passed a wrong URI type.
        """
        ## Arrange
        invalid_uris = [
            ([1], "list"),
            ([1, 'Hi'], "list with mixed types"),
            ({'key': 1, 'value': 'Hi'}, "dictionary"),
            ({1, 'Hi'}, "set"),
            (1, "integer"),
            (3.14, "float"),
            (None, "NoneType"),
            (True, "boolean")
        ]

        ## Act and Assert
        # Run through each uri
        for uri, description in invalid_uris:
            with self.subTest(uri_type=description):
                with self.assertRaises(Exception):
                    MilvusClientInit(uri=uri)


    ## Test error handling bad client
    @patch('pyfiles.milvus_utils.MilvusClient')
    def test_init_bad_client(
        self, 
        mock_client
    ):
        """
        Test error handling of `MilvusClientInit` when passed a wrong client type.
        
        Verifications
        ------------
            Initializing the `MilvusClientInit` raises an exception when passed a wrong client type.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when `MilvusClientInit` is passed a wrong client type.
        """
        ## Arrange
        invalid_clients = [
            ([1], "list"),
            ([1, 'Hi'], "list with mixed types"),
            ({'key': 1, 'value': 'Hi'}, "dictionary"),
            ({1, 'Hi'}, "set"),
            (1, "integer"),
            (3.14, "float"),
            (None, "NoneType"),
            (True, "boolean")
        ]

        ## Act and Assert
        # Run through each client
        for client, description in invalid_clients:
            with self.subTest(client_type=description):
                # MilvusClient | Create a mock instance of the client
                mock_client_instance = MagicMock(spec=client)
                mock_client.return_value = mock_client_instance
                with self.assertRaises(Exception):
                    MilvusClientInit(uri=uri, client=mock_client)


    ## Test successful creation of field
    def test_create_field_success(self):
        """
        Test successfully creating a field.
        
        Verifications
        ------------
            The correct method is called with the correct parameters.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `schema` (CollectionSchema) is mocked and returns a mock instance.
        
        Asserts
        ------------
            The `add_field` method of the collection schema is called exactly once with the correct parameters.
        """
        ## Arrange
        params = field_params_list[0]
        # schema | Create a mock instance of the collection schema
        schema = MagicMock(spec=CollectionSchema)
        # MilvusClient | Create a mock instance of the client
        mock_client = MagicMock(spec=MilvusClient)
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)
        client._create_field(schema=schema, params=params)

        ## Assert
        schema.add_field.assert_called_once_with(**params)


    ## Test error handling bad schema or params type
    def test_create_field_bad_args(self):
        """
        Test error handling of creating a field when the method is passed wrong argument types.
        
        Verifications
        ------------
            Invoking the method raises an exception when passed wrong argument types.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `schema` (CollectionSchema) is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when the `_create_field` method is passed wrong argument types.
        """
        ## Arrange
        invalid_schema = [
            ([1], "list"),
            ([1, 'Hi'], "list with mixed types"),
            ({1, 'Hi'}, "set"),
            (1, "integer"),
            (3.14, "float"),
            (None, "NoneType"),
            (True, "boolean")
        ]
        invalid_params = invalid_schema
        valid_params = field_params_list[0]
        # schema | Create a mock instance of a valid collection schema
        valid_schema = MagicMock(spec=CollectionSchema)
        # MilvusClient | Create a mock instance of the client
        mock_client = MagicMock(spec=MilvusClient)        

        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Act and Assert
        # Run through each invalid schema with valid params
        method_args = {"params": valid_params}
        self._loop_through_params(
            param_name='schema', 
            param_list=invalid_schema, 
            method_name='_create_field', 
            method_args=method_args, 
            client=client
        )
        # Run through each invalid params with valid schema
        method_args = {"schema": valid_schema}
        self._loop_through_params(
            param_name='params', 
            param_list=invalid_params, 
            method_name='_create_field', 
            method_args=method_args, 
            client=client
        )

    
    ## Test failed creation of field
    def test_create_field_failure(self):
        """
        Test failed creation of a field.
        
        Verifications
        ------------
            Invoking the method raises an exception when a failure is introduced.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `schema` (CollectionSchema) is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when the `add_field` method fails.
        """
        ## Arrange
        params = field_params_list[0]
        # schema | Create a mock instance of the collection schema
        schema = MagicMock(spec=CollectionSchema)
        schema.add_field.side_effect = Exception
        # MilvusClient | Create a mock instance of the client
        mock_client = MagicMock(spec=MilvusClient)
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Assert
        with self.assertRaises(Exception):
            client._create_field(schema=schema, params=params)


    ## Test successful creation of index
    def test_create_index_success(self):
        """
        Test successfully creating an index.
        
        Verifications
        ------------
            The correct method is called with the correct parameters.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `index_params` (IndexParams) is mocked and returns a mock instance.
        
        Asserts
        ------------
            The `add_index` method of the collection schema is called exactly once with the correct parameters.
        """
        ## Arrange
        params = index_params_list[0]
        # index_params | Create a mock instance of the collection index parameters
        index_params = MagicMock(spec=IndexParams)
        # MilvusClient | Create a mock instance of the client
        mock_client = MagicMock(spec=MilvusClient)
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)
        client._create_index(index_params=index_params, params=params)

        ## Assert
        index_params.add_index.assert_called_once_with(**params)


    ## Test error handling bad schema or params type
    def test_create_index_bad_args(self):
        """
        Test error handling of creating an index when the method is passed wrong argument types.
        
        Verifications
        ------------
            Invoking the method raises an exception when passed wrong argument types.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `index_params` (IndexParams) is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when the `_create_index` method is passed wrong argument types.
        """
        ## Arrange
        invalid_index_params = [
            ([1], "list"),
            ([1, 'Hi'], "list with mixed types"),
            ({1, 'Hi'}, "set"),
            (1, "integer"),
            (3.14, "float"),
            (None, "NoneType"),
            (True, "boolean")
        ]
        invalid_params = invalid_index_params
        valid_params = index_params_list[0]
        # index_params | Create a mock instance of valid collection index parameters
        valid_index_params = MagicMock(spec=IndexParams)
        # MilvusClient | Create a mock instance of the client
        mock_client = MagicMock(spec=MilvusClient)        

        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Act and Assert
        # Run through each invalid index_params with valid params
        method_args = {"params": valid_params}
        self._loop_through_params(
            param_name='index_params', 
            param_list=invalid_index_params, 
            method_name='_create_index', 
            method_args=method_args, 
            client=client
        )
        # Run through each invalid params with valid index_params
        method_args = {"index_params": valid_index_params}
        self._loop_through_params(
            param_name='params', 
            param_list=invalid_params, 
            method_name='_create_index', 
            method_args=method_args, 
            client=client
        )


    ## Test failed creation of index
    def test_create_index_failure(self):
        """
        Test failed creation of an index.
        
        Verifications
        ------------
            Invoking the method raises an exception when a failure is introduced.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `index_params` (IndexParams) is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when the `add_index` method fails.
        """
        ## Arrange
        params = index_params_list[0]
        # index_params | Create a mock instance of the collection index parameters
        index_params = MagicMock(spec=IndexParams)
        index_params.add_index.side_effect = Exception
        # MilvusClient | Create a mock instance of the client
        mock_client = MagicMock(spec=MilvusClient)
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Assert
        with self.assertRaises(Exception):
            client._create_index(index_params=index_params, params=params)


    ## Test successful collection listing
    @patch('pyfiles.milvus_utils.MilvusClient.list_collections')
    def test_list_collection_success(
        self, 
        mock_list
    ):
        """
        Test successfully listing collections.
        
        Verifications
        ------------
            The correct response is obtained from invoking the correct method.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `MilvusClient.list_collections` is mocked and returns a mock instance.
        
        Asserts
        ------------
            The `MilvusClient.list_collections` method is called exactly once.
            A list of strings is obtained from invoking the `MilvusClient.list_collections`.
        """
        ## Arrange
        collections_ex = ['collection-0', 'collection-1']
        # MilvusClient | Create a mock instance of the client
        mock_client = MagicMock(spec=MilvusClient)
        # MilvusClient.list_collections | Create a mock instance
        mock_list.return_value = collections_ex
        mock_client.list_collections = mock_list

        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)
        collections = client.list_collections()

        ## Assert
        mock_list.assert_called_once()
        self.assertEqual(collections, collections_ex)

    
    ## Test error handling bad collection list type
    def test_list_collection_bad_list(self):
        """
        Test error handling of list collections method when result has wrong type.
        
        Verifications
        ------------
            Getting a wrong result type raises an exception.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when `MilvusClient.list_collections` results in a wrong collections type.
        """
        ## Arrange
        invalid_collection_lists = [
            ([1], "list of integers"),
            ([1, 'Hi'], "list with mixed types"),
            ({'key': 1, 'value': 'Hi'}, "dictionary"),
            ({1, 'Hi'}, "set"),
            (1, "integer"),
            (3.14, "float"),
            (None, "NoneType"),
            (True, "boolean")
        ]
        # MilvusClient | Create a mock instance of the client
        mock_client = MagicMock(spec=MilvusClient)

        ## Act
        MilvusClientInit(uri=uri, client=mock_client)

        ## Assert
        # Run through each collection list
        for collection_list, description in invalid_collection_lists:
            with self.subTest(collection_list_type=description):
                with self.assertRaises(Exception):
                    mock_list.return_value = collection_list
                    mock_client.list_collections = mock_list


    ## Test failed collection listing
    @patch('pyfiles.milvus_utils.MilvusClient.list_collections')
    def test_list_collection_failure(
        self, 
        mock_list
    ):
        """
        Test failed collection listing.
        
        Verifications
        ------------
            Invoking the method raises an exception when a failure is introduced.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when the `client.list_collections` method fails.
        """
        ## Arrange
        # MilvusClient | Create a mock instance of the client
        mock_client = MagicMock(spec=MilvusClient)
        mock_client.list_collections.side_effect = Exception

        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)
        
        ## Assert
        with self.assertRaises(Exception):
            collections = client.list_collections()


    ## Test successful collection creation
    @patch('pyfiles.milvus_utils.MilvusClientInit._create_index')
    @patch('pyfiles.milvus_utils.MilvusClientInit._create_field')
    @patch('pyfiles.milvus_utils.MilvusClient.create_collection')
    @patch('pyfiles.milvus_utils.MilvusClient.prepare_index_params')
    @patch('pyfiles.milvus_utils.MilvusClient.create_schema')
    def test_create_collection_success(
        self, 
        mock_create_schema, 
        mock_prepare_index_params, 
        mock_create_collection, 
        mock_create_field, 
        mock_create_index
    ):
        """
        Test successfully creating a collection.
        
        Verifications
        ------------
            The correct methods are invoked with expected calls.

        Mocks
        ------------
            `MilvusClient` to mock client calls
            `MilvusClient.create_schema` to mock creating schema
            `CollectionSchema` to mock return value of `create_schema`
            `MilvusClient.prepare_index_params` to mock creating index params
            `IndexParams` to mock return value of `prepare_index_params`
            `MilvusClient.list_collections` to mock listing collections
            `MilvusClient.create_collection` to mock creating a collection
            `MilvusClientInit._create_field` to mock adding fields to schema
            `MilvusClientInit._create_index` to mock adding indices to index params
        
        Asserts
        ------------
            `MilvusClient.create_schema` called once with the correct arguments.
            `MilvusClientInit._create_field` called multiple times with the correct field parameters.
            `schema.add_function` called once with the correct arguments.
            `MilvusClient.prepare_index_params` called once with the correct arguments.
            `MilvusClientInit._create_index` called multiple times with the correct index parameters.
            `MilvusClient.create_collection` called once with the correct arguments.
            `MilvusClient.list_collections` called twice.
        """
        
        ## Arrange
        # schema | Create a mock schema instance
        mock_schema = MagicMock(spec=CollectionSchema)
        mock_create_schema.return_value = mock_schema
        # index_params | Create a mock index params
        mock_index_params = MagicMock(spec=IndexParams)
        mock_prepare_index_params.return_value = mock_index_params
        # list_collections | Mock listing collections
        mock_list = MagicMock()
        mock_list.side_effect = [[], [collection_name]]
        
        # MilvusClient | Create a client instance with mocked dependencies
        mock_client = MagicMock(spec=MilvusClient)
        mock_client.create_schema = mock_create_schema
        mock_client.prepare_index_params = mock_prepare_index_params
        mock_client.create_collection = mock_create_collection
        mock_client.list_collections = mock_list
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)
        client.create_collection(
            name=collection_name, 
            field_params_list=field_params_list, 
            func_list=[func_bm25], 
            index_params_list=index_params_list,
        )

        ## Assert
        mock_create_schema.assert_called_once_with(enable_dynamic_field=True)
        expected_field_calls = [call(mock_schema, field_params) for field_params in field_params_list]
        mock_create_field.assert_has_calls(expected_field_calls)
        mock_schema.add_function.assert_called_once_with(func_bm25)
        
        mock_prepare_index_params.assert_called_once()
        expected_index_calls = [call(mock_index_params, index_params) for index_params in index_params_list]
        mock_create_index.assert_has_calls(expected_index_calls)

        mock_create_collection.assert_called_once_with(
            collection_name=collection_name,
            schema=mock_schema,
            index_params=mock_index_params
        )

        expected_list_calls = [call()]*2
        mock_list.assert_has_calls(expected_list_calls)


    ## Test error handling for bad create collection arguments
    def test_create_collection_bad_args(self):
        """
        Test error handling of creaing a collection when passed arguments with the wrong types.
        
        Verifications
        ------------
            Invoking the method raises an exception when passed wrong argument types.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when `MilvusClientInit.create_collections` is pass the wrong argument types.
        """
        
        ## Arrange
        # name | Create valid and invalid instances
        invalid_name = [
            ([1, 2], "list of integers"),
            ([1, 'Hi'], "list with mixed types"),
            ({1, 'Hi'}, "set"),
            (1, "integer"),
            (3.14, "float"),
            (None, "NoneType"),
            (True, "boolean")
        ]
        valid_name = collection_name

        # field_params_list | Create valid and invalid instances
        invalid_field_params_list = invalid_name
        valid_field_params_list = field_params_list

        # field_params_list | Create valid and invalid instances
        invalid_func_list = invalid_name
        valid_func_list = [func_bm25]

        # index_params_list | Create valid and invalid instances
        invalid_index_params_list = invalid_name
        valid_index_params_list = index_params_list

        # MilvusClient | Create a client instance
        mock_client = MagicMock(spec=MilvusClient)
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Assert
        # Run through each invalid name
        method_args = {
            "field_params_list": valid_field_params_list,
            "func_list": valid_func_list,
            "index_params_list": valid_index_params_list
        }
        self._loop_through_params(
            param_name='name', 
            param_list=invalid_name, 
            method_name='create_collection', 
            method_args=method_args, 
            client=client
        )

        # Run through each invalid field_params_list
        method_args = {
            "name": valid_name,
            "func_list": valid_func_list,
            "index_params_list": valid_index_params_list
        }
        self._loop_through_params(
            param_name='field_params_list', 
            param_list=invalid_field_params_list, 
            method_name='create_collection', 
            method_args=method_args, 
            client=client
        )

        # Run through each invalid func_list
        method_args = {
            "name": valid_name,
            "field_params_list": valid_field_params_list,
            "index_params_list": valid_index_params_list
        }
        self._loop_through_params(
            param_name='func_list', 
            param_list=invalid_func_list, 
            method_name='create_collection', 
            method_args=method_args, 
            client=client
        )

        # Run through each invalid index_params_list
        method_args = {
            "name": valid_name,
            "field_params_list": valid_field_params_list,
            "func_list": valid_func_list
        }
        self._loop_through_params(
            param_name='index_params_list', 
            param_list=invalid_index_params_list, 
            method_name='create_collection', 
            method_args=method_args, 
            client=client
        )

    
    ## Test successful dropping of collection
    @patch('pyfiles.milvus_utils.MilvusClient.drop_collection')
    def test_drop_collection_success(
        self, 
        mock_drop_collection
    ):
        """
        Test successfully dropping a collection.
        
        Verifications
        ------------
            The correct method is called with the correct parameters.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `MilvusClient.list_collections` is mocked and returns a mock instance.
            `MilvusClient.drop_collection` is mocked
        
        Asserts
        ------------
            The `MilvusClient.drop_collection` method is called exactly once with the correct parameters.
            The `MilvusClient.list_collections` method is called twice.
        """
        ## Arrange
        # list_collections | Mock listing collections
        mock_list = MagicMock()
        mock_list.side_effect = [[collection_name], []]
        
        # MilvusClient | Create a client instance with mocked dependencies
        mock_client = MagicMock(spec=MilvusClient)
        mock_client.drop_collection = mock_drop_collection
        mock_client.list_collections = mock_list
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)
        client.drop_collection(name=collection_name)

        ## Assert
        mock_client.drop_collection.assert_called_once_with(collection_name=collection_name)

        expected_list_calls = [call()]*2
        mock_list.assert_has_calls(expected_list_calls)


    ## Test error handling for bad drop collection arguments
    def test_drop_collection_bad_args(self):
        """
        Test error handling of dropping a collection when passed arguments with the wrong types.
        
        Verifications
        ------------
            Invoking the method raises an exception when passed wrong argument types.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when `MilvusClientInit.drop_collections` is passed the wrong argument types.
        """
        
        ## Arrange
        # name | Create valid and invalid instances
        invalid_name = [
            ([1, 2], "list of integers"),
            ([1, 'Hi'], "list with mixed types"),
            ({1, 'Hi'}, "set"),
            (1, "integer"),
            (3.14, "float"),
            (None, "NoneType"),
            (True, "boolean")
        ]

        # MilvusClient | Create a client instance
        mock_client = MagicMock(spec=MilvusClient)
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Assert
        # Run through each invalid name
        method_args = {}
        self._loop_through_params(
            param_name='name', 
            param_list=invalid_name, 
            method_name='drop_collection', 
            method_args=method_args, 
            client=client
        )


    ## Test failed dropping of collection
    @patch('pyfiles.milvus_utils.MilvusClient.drop_collection')
    def test_drop_collection_failure(
        self, 
        mock_drop_collection
    ):
        """
        Test failed collection dropping.
        
        Verifications
        ------------
            Invoking the method raises an exception when a failure is introduced.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `MilvusClient.list_collections` is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when the `client.drop_collection` method fails.
        """
        ## Arrange
        # list_collections | Mock listing collections
        mock_list = MagicMock()
        mock_list.return_value = [collection_name]
        
        # MilvusClient | Create a client instance with mocked dependencies
        mock_client = MagicMock(spec=MilvusClient)
        mock_client.drop_collection.side_effect = Exception
        mock_client.list_collections = mock_list
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Assert
        with self.assertRaises(Exception):
            client.drop_collection(name=collection_name)


    ## Test successful inserting of data
    @patch('pyfiles.milvus_utils.MilvusClient.insert')
    def test_insert_success(
        self, 
        mock_insert
    ):
        """
        Test successfully inserting data into a collection.
        
        Verifications
        ------------
            The correct method is called with the correct parameters.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `MilvusClient.insert` is mocked and returns a mock instance.
        
        Asserts
        ------------
            The `MilvusClient.insert` method is called exactly once with the correct parameters.
            `time.sleep` is called once with the correct parameters.
            The `MilvusClient.list_collections` method is called once.
        """
        ## Arrange        
        # MilvusClient | Create a client instance with mocked dependencies
        mock_insert.return_value = {}
        mock_client = MagicMock(spec=MilvusClient)
        mock_client.insert = mock_insert
        # list_collections | Mock listing collections
        mock_list = MagicMock()
        mock_list.return_value = [collection_name]
        mock_client.list_collections = mock_list
        
        ## Act and Assert
        with patch('pyfiles.milvus_utils.time.sleep') as mock_sleep:
            client = MilvusClientInit(uri=uri, client=mock_client)
            client.insert(name=collection_name, data=data_ex)

            # Verify insert is called
            mock_client.insert.assert_called_once_with(
                collection_name=collection_name,
                data=data_ex
            )

            # Verify sleep is called once with the correct argument
            mock_sleep.assert_called_once_with(0.5)
        
        mock_list.assert_called_once()

    
    ## Test error handling for bad insert arguments
    def test_insert_bad_args(self):
        """
        Test error handling of inserting data when passed arguments with the wrong types.
        
        Verifications
        ------------
            Invoking the method raises an exception when passed wrong argument types.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when `MilvusClientInit.insert` is passed the wrong argument types.
        """
        
        ## Arrange
        # name | Create valid and invalid instances
        invalid_name = [
            ([1, 2], "list of integers"),
            ([1, 'Hi'], "list with mixed types"),
            ({1, 'Hi'}, "set"),
            (1, "integer"),
            (3.14, "float"),
            (None, "NoneType"),
            (True, "boolean")
        ]
        valid_name = collection_name

        invalid_data = invalid_name
        valid_data = data_ex

        # MilvusClient | Create a client instance
        mock_client = MagicMock(spec=MilvusClient)
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Assert
        # Run through each invalid name
        method_args = {"data": valid_data}
        self._loop_through_params(
            param_name='name', 
            param_list=invalid_name, 
            method_name='insert', 
            method_args=method_args, 
            client=client
        )

        # Run through each invalid data
        method_args = {"name": valid_name}
        self._loop_through_params(
            param_name='data', 
            param_list=invalid_data, 
            method_name='insert', 
            method_args=method_args, 
            client=client
        )


    ## Test failed inserting of data
    @patch('pyfiles.milvus_utils.MilvusClient.insert')
    def test_insert_failure(
        self, 
        mock_insert
    ):
        """
        Test failed data insert.
        
        Verifications
        ------------
            Invoking the method raises an exception when a failure is introduced.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `MilvusClient.list_collections` is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when the `client.insert` method fails.
        """
        ## Arrange        
        # MilvusClient | Create a client instance with mocked dependencies
        mock_client = MagicMock(spec=MilvusClient)
        mock_client.insert.side_effect = Exception
        # list_collections | Mock listing collections
        mock_list = MagicMock()
        mock_list.return_value = [collection_name]
        mock_client.list_collections = mock_list
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Assert
        with self.assertRaises(Exception):
            client.insert(name=collection_name, data=data_ex)


    ## Test successful full text search
    @patch('pyfiles.milvus_utils.MilvusClient.search')
    def test_full_text_search_success(
        self, 
        mock_search
    ):
        """
        Test successfully performing a full text search.
        
        Verifications
        ------------
            The correct method is called with the correct parameters.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
            `SearchResult` is mocked as the return value for the `MilvusClient.search` method.
            `HybridHits` is mocked to populate the `SearchResult` return value.
            `Hit` is mocked to populate the `HybridHits` return value.
        
        Asserts
        ------------
            The `search` method of the `MilvusClient` is called once with the correct parameters.
            The `MilvusClient.list_collections` method is called once.
        """
        ## Arrange        
        anns_field = 'sparse'
        output_fields = ['text']
        search_params = {
            'params': {'drop_ratio_search': 0.2},
        }

        # Results | Create a mock result
        # SearchResult is list of HybridHits
        # HybridHits is list of Hit
        mock_hit = MagicMock(spec=Hit)
        mock_hybrid_hits = MagicMock(spec=HybridHits)
        mock_hybrid_hits.return_value = [mock_hit]
        mock_search_result = MagicMock(spec=SearchResult)
        mock_search_result.return_value = [mock_hybrid_hits]
        
        # MilvusClient | Create a client instance with mocked dependencies
        mock_client = MagicMock(spec=MilvusClient)
        mock_search.return_value = mock_search_result
        mock_client.search = mock_search

        # list_collections | Mock listing collections
        mock_list = MagicMock()
        mock_list.return_value = [collection_name]
        mock_client.list_collections = mock_list
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)
        client.full_text_search(
            name = collection_name, 
            query_list = query_list, 
            limit = lim_results
        )

        ## Assert
        mock_client.search.assert_called_once_with(
            collection_name=collection_name, 
            data=query_list,
            anns_field=anns_field,
            output_fields=output_fields,
            limit=lim_results,
            search_params=search_params
        )
        mock_list.assert_called_once()


    ## Test error handling for bad full text search arguments
    def test_full_text_search_bad_args(self):
        """
        Test error handling of performing a full text search when passed arguments with the wrong types.
        
        Verifications
        ------------
            Invoking the method raises an exception when passed wrong argument types.
            Exception is propagated correctly.

        Mocks
        ------------
            `MilvusClient` is mocked and returns a mock instance.
        
        Asserts
        ------------
            Exception is raised when `MilvusClientInit.full_text_search` is passed the wrong argument types.
            The `MilvusClient.list_collections` method is called once.
        """
        
        ## Arrange
        # name | Create valid and invalid instances
        invalid_name = [
            ([1, 2], "list of integers"),
            ([1, 'Hi'], "list with mixed types"),
            ({1, 'Hi'}, "set"),
            (3.14, "float"),
            (None, "NoneType"),
            (True, "boolean")
        ]
        valid_name = collection_name

        invalid_query_list = invalid_name
        valid_query_list = query_list

        invalid_limit = invalid_name
        valid_limit = lim_results

        # MilvusClient | Create a client instance
        mock_client = MagicMock(spec=MilvusClient)
        # list_collections | Mock listing collections
        mock_list = MagicMock()
        mock_list.return_value = [collection_name]
        mock_client.list_collections = mock_list
        
        ## Act
        client = MilvusClientInit(uri=uri, client=mock_client)

        ## Assert
        # Run through each invalid name
        method_args = {"query_list": valid_query_list, "limit": valid_limit}
        self._loop_through_params(
            param_name='name', 
            param_list=invalid_name, 
            method_name='full_text_search', 
            method_args=method_args, 
            client=client
        )

        # Run through each invalid data
        method_args = {"name": valid_name, "limit": valid_limit}
        self._loop_through_params(
            param_name='query_list', 
            param_list=invalid_query_list, 
            method_name='full_text_search', 
            method_args=method_args, 
            client=client
        )

        # Run through each invalid limit
        method_args = {"name": valid_name, "query_list": valid_query_list}
        self._loop_through_params(
            param_name='limit', 
            param_list=invalid_limit, 
            method_name='full_text_search', 
            method_args=method_args, 
            client=client
        )

        mock_list.assert_called_once()