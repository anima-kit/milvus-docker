### milvus_utils
## Defines functions needed to setup, manange, and query a vector database with a Milvus client.
## Based on Milvus docs | https://milvus.io/docs/full-text-search.md

## Imports
# Third-party modules
import time
import pprint

from pymilvus.client.search_result import SearchResult # type: ignore
from pymilvus.milvus_client.index import IndexParams # type: ignore
from pymilvus import ( # type: ignore
    MilvusClient, 
    CollectionSchema, 
    Function, 
    FunctionType, 
    DataType 
)

from typing import List

## Internal modules
from validators import milvus_types
from pyfiles.logger import (
    logger, 
    with_spinner
)


## Constants
# URI | Milvus server uri
uri: str = 'http://localhost:19530'

## Index params list
# List of dictionaries defining the indices to add to index params
# Here, we're only working with sparse vectors created with BM25
index_params_list: List[dict] = [
    {
        "field_name": "sparse",
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
        "params": {
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 3,   # Maximize importance of term frequency
            "bm25_b": 1     # Full normalization of docs
        }
    }
]

## BM25 embed function
# The function to get sparse embeddings from text
func_bm25: Function = Function(
    name="text_bm25_emb",
    input_field_names=["text"],
    output_field_names=["sparse"],
    function_type=FunctionType.BM25,
)

## Field params list
# List of dictionaries defining the fields to add to the schema
# This is how we describe our data:
# just need text and sparse vectors for this demo (full-text search only)
field_params_list: List[dict] = [
    {
        "field_name": "id", 
        "datatype": DataType.INT64, 
        "is_primary": True, 
        "auto_id": True
    },
    {
        "field_name": "text", 
        "datatype": DataType.VARCHAR, 
        "max_length": 1000, 
        "enable_analyzer": True
    },
    {
        "field_name": "sparse", 
        "datatype": DataType.SPARSE_FLOAT_VECTOR
    }
]

## Collection name
collection_name: str = 'collection_ex'

## Example data
# Default data to use for the insert method
data_ex: List[dict] = [
    {'text': 'information retrieval is a field of study.'},
    {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
    {'text': 'data mining and information retrieval overlap in research.'},
    {'text': 'the rest of the lyrics go,'},
    {'text': 'Last night I dreamed about'}
]

## Query list
# Default list of queries to search the database
query_list: List[str] = ["What's the focus of information retrieval?"]

## Result limit
# Default maximum number of results to get from search
lim_results: int = 3



class MilvusClientInit:
    """
    A Milvus client that can be used to manage and search data.

    The user can do a full-text search on their data by initializing the client then using the `full_text_search` method to get results.

    For example, to initialize the client for a given URI:
    ```python
    uri = 'http://localhost:19530'
    client = MilvusClientInit(uri=uri)
    ```

    Then, to create a collection:
    ```python
    name = 'collection_ex'

    field_params_list = [
        {
            "field_name": "id", 
            "datatype": DataType.INT64, 
            "is_primary": True, 
            "auto_id": True
        },
        {
            "field_name": "text", 
            "datatype": DataType.VARCHAR, 
            "max_length": 1000, 
            "enable_analyzer": True
        },
        {
            "field_name": "sparse", 
            "datatype": DataType.SPARSE_FLOAT_VECTOR
        }
    ]

    func_list = [
        Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    ]

    index_params_list = [
        {
            "field_name": "sparse",
            "index_type": "AUTOINDEX",
            "metric_type": "BM25",
        }
    ]

    client.create_collection(
        name=name, 
        field_params_list=field_params_list, 
        func_list=func_list, 
        index_params_list=index_params_list
    )
    ```

    Then, to insert some data:
    ```python
    data = [
        {'text': 'information retrieval is a field of study.'},
        {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
        {'text': 'data mining and information retrieval overlap in research.'},
        {'text': 'the rest of the lyrics go,'},
        {'text': 'Last night I dreamed about'}
    ]
    client.insert(name=name, data=data)
    ```

    Finally, to do a full-text search:
    ```python
    query_list = ['whats the focus of information retrieval?']
    lim_results = 3
    client.full_text_search(name=name, query_list=query_list, limit=limit)
    ```
    
    Attributes
    ------------
        uri: str, Optional
            The uri on which to host the Milvus client.
            Defaults to 'http://localhost:19530'.
        client: MilvusClient
            The Milvus client to use to manage and query data.
    """
    def __init__(
        self, 
        uri: str = uri,
        client: MilvusClient | None = None
    ) -> None:
        """
        Initialize the Milvus client hosted on the given URI.
        
        Args
        ------------
            uri: str, Optional
                The uri on which to host the Milvus client.
                Defaults to 'http://localhost:19530'.
            
        Raises
        ------------
            Exception: 
                If initialization fails, error is logged and raised.
        """
        ## Validate all argument types
        milvus_types.InitClientParams(
            uri = uri
        )

        ## Start Init
        try:
            self.uri = uri
            # Initialize MilvusClient from PyMilvus
            if client is None:
                self.client: MilvusClient = self._init_client()
            else:
                ## Validate client
                milvus_types.InitClientResults(
                    results=client
                )
                self.client = client
        except Exception as e:
            logger.info(f'❌ Problem initializing Milvus client: {str(e)}')
            raise


    # Initialize MilvusClient
    def _init_client(
        self
    ) -> MilvusClient:
        """
        Connect the Milvus client.
        
        Returns
        ------------
            MilvusClient
                The client instance.
            
        Raises
        ------------
            Exception: 
                If client connection fails, error is logged and raised.
        """
        logger.info(f'⚙️ Starting Milvus client on URI `{self.uri}`')
        try:
            ## Define MilvusClient with PyMilvus library
            client: MilvusClient = MilvusClient(
                uri=self.uri
            )
            
            ## Validate results
            milvus_types.InitClientResults(
                results=client
            )

            ## Return results
            logger.info(f'⚙️ Milvus client connected at `{self.uri}`')
            return client
        except Exception as e:
            logger.error(f'❌ Problem connecting to Milvus client: `{str(e)}`')
            raise


    # Create field for schema
    def _create_field(
        self, 
        schema: CollectionSchema, 
        params: dict
    ) -> None:
        """
        Add a field with the given parameters to the given schema.

        Args
        ------------
            schema: CollectionSchema
                The schema for which a field will be created.
                The fields will be created using the schema.add_field method.
            params: dict
                Dictionary with key-value pairs given by the argument-value pairs for the schema.add_field method.
            
        Raises
        ------------
            Exception: 
                If adding the field fails, error is logged and raised.
        """
        ## Validate all argument types
        milvus_types.CreateFieldParams(
            collection_schema=schema,
            params=params
        )

        ## Start Create Field
        logger.info(f'⚙️ Creating field for params: \n {pprint.pformat(params, indent=0, width=500)} \n')
        try:
            # Add the field to the schema for the given params
            schema.add_field(**params)
            logger.info(f'✅ Field added to schema')
        except Exception as e:
            logger.info(f'❌ Problem creating field: `{str(e)}`')
            raise

    
    ## Create index for index parameters
    def _create_index(
        self, 
        index_params: IndexParams, 
        params: dict
    ) -> None:
        """
        Add an index with the given parameters to the given list of index parameters.

        Args
        ------------
            index_params: IndexParams
                The index parameters for which an index will be created.
                The index will be created using the index_params.add_index method.
            params: dict
                Dictionary with key-value pairs given by the argument-value pairs for the index_params.add_index method.
            
        Raises
        ------------
            Exception: 
                If adding the index fails, error is logged and raised.
        """
        ## Validate all argument types
        milvus_types.CreateIndexParams(
            index_params=index_params,
            params=params
        )

        ## Start Create Index
        logger.info(f'⚙️ Creating index for params: \n {pprint.pformat(params, indent=0, width=500)} \n')
        try:
            # Add the index to the index params for the given params
            index_params.add_index(**params)
            logger.info(f'✅ Index added to index params')
        except Exception as e:
            logger.info(f'❌ Problem creating index params: `{str(e)}`')
            raise


    ## List all client collections
    def list_collections(
        self
    ) -> List[str | None]:
        """
        Get all collections for the client.

        For example, to get the collections for a given client:
        ```python
        # Initialize the client
        uri = 'http://localhost:19530'
        client = MilvusClientInit(uri=uri)

        # List collections
        client.list_collections()
        ```

        Returns
        ------------
            List[str | None]: 
                A list of the names of all available collections.
            
        Raises
        ------------
            Exception: 
                If getting the collections fails, error is logged and raised.
        """
        try:
            ## List all client collections
            collections = self.client.list_collections()

            ## Validate results
            milvus_types.ListCollectionsResults(
                results=collections
            )

            ## Return results
            logger.info(f'📝 Available collections: \n {pprint.pformat(collections, indent=0, width=500)} \n')
            return collections
        except Exception as e:
            logger.error(f'❌ Problem listing collections: `{str(e)}`')
            raise   


    ## Create a collection for the client
    def create_collection(
        self, 
        name: str = collection_name, 
        field_params_list: List[dict] = field_params_list, 
        func_list: List[Function] = [func_bm25], 
        index_params_list: List[dict] = index_params_list,
    ) -> None:
        """
        Create a collection with the given name, field parameters, embedding functions, and index parameters.

        For example, one can create a collection to describe data with text and sparse vectors. This will be good for performing full-text searches:
        ```python
        # Initialize the client
        uri = 'http://localhost:19530'
        client = MilvusClientInit(uri=uri)

        # Create a collection
        name = 'collection_ex'

        field_params_list = [
            {
                "field_name": "id", 
                "datatype": DataType.INT64, 
                "is_primary": True, 
                "auto_id": True
            },
            {
                "field_name": "text", 
                "datatype": DataType.VARCHAR, 
                "max_length": 1000, 
                "enable_analyzer": True
            },
            {
                "field_name": "sparse", 
                "datatype": DataType.SPARSE_FLOAT_VECTOR
            }
        ]

        func_list = [
            Function(
                name="text_bm25_emb",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        ]

        index_params_list = [
            {
                "field_name": "sparse",
                "index_type": "AUTOINDEX",
                "metric_type": "BM25",
            }
        ]

        client.create_collection(
            name=name, 
            field_params_list=field_params_list, 
            func_list=func_list, 
            index_params_list=index_params_list
        )
        ```

        Args
        ------------
            name: str, Optional
                The name of the collection to create.
                Defaults to 'collection_ex'.
            field_params_list: List[dict]
                List of dictionaries for the field parameters.
                Fields to add to the collection (to describe data).
                Key-value pairs give the argument-value pairs for the schema.add_field method
                Defaults to: [
                    {
                        "field_name": "id", 
                        "datatype": DataType.INT64, 
                        "is_primary": True, 
                        "auto_id": True
                    },
                    {
                        "field_name": "text", 
                        "datatype": DataType.VARCHAR, 
                        "max_length": 1000, 
                        "enable_analyzer": True
                    },
                    {
                        "field_name": "sparse", 
                        "datatype": DataType.SPARSE_FLOAT_VECTOR
                    }
                ]
            index_params_list: List[dict]
                List of dictionaries for the index parameters.
                Fields to use in search with given specifications.
                Key-value pairs give the argument-value pairs for the index_params.add_index method
                Defaults to: [
                    {
                        "field_name": "sparse",
                        "index_type": "AUTOINDEX",
                        "metric_type": "BM25",
                    }
                ]
            func_list: List[Function]
                List of functions for the data embedding.
                Defaults to the BM25 embed function:
                    Function(
                        name="text_bm25_emb",
                        input_field_names=["text"],
                        output_field_names=["sparse"],
                        function_type=FunctionType.BM25,
                    ) 
            
        Raises
        ------------
            Exception: 
                If creating the collection fails, error is logged and raised.
        """
        ## Validate all argument types
        milvus_types.CreateCollectionParams(
            name=name,
            field_params_list=field_params_list,
            func_list=func_list,
            index_params_list=index_params_list
        )

        ## Check if collection already exists
        collections: List[str | None] = self.list_collections()
        if name in collections:
            logger.info(f'✅ Collections `{name}` already exists.')
            return

        ## Start Create Collection
        logger.info(f'⚙️ Creating collection `{name}`')
        try:
            ## Initialize the collection schema
            # Allow for adding different fields later on with enable_dynamic_field
            schema: CollectionSchema = self.client.create_schema(
                enable_dynamic_field=True,
            )

            ## Add all fields to the schema
            # This is how to describe the data
            for params in field_params_list:
                self._create_field(schema, params)

            ## Add all functions to the schema (embedding)
            # This is how to represent the data
            for func in func_list:
                schema.add_function(func)

            ## Add all indices to the index params
            # This is how to search the data
            index_params: IndexParams = self.client.prepare_index_params()
            for params in index_params_list:
                self._create_index(index_params, params)

            ## Create collection
            self.client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params
            )

            ## List collections
            # Make sure collection was added properly
            collections = self.list_collections()
            if name in collections:
                logger.info(f'✅ Created collection `{name}`')
            else:
                error_message = f"The new collection `name` is not in the collections list."
                logger.error(error_message)
                raise ValueError(error_message)
        except Exception as e:
            logger.error(f'❌ Problem creating collection `{name}`: `{str(e)}`')
            raise   


    ## Drop a collection for the client
    def drop_collection(
        self, 
        name: str = collection_name
    ) -> None:
        """
        Drop a collection with the given name.

        For example, to create then drop a collection:
        ```python
        # Initialize the client
        uri = 'http://localhost:19530'
        client = MilvusClientInit(uri=uri)

        # Create a collection
        name = 'collection_ex'
        client.create_collection(name=name)

        # Drop the colleciton
        client.drop_collection(name=name)
        ```

        Args
        ------------
            name: str
                The name of the collection to drop.
            
        Raises
        ------------
            Exception: 
                If dropping the collection fails, error is logged and raised.
        """
        ## Validate all argument types
        milvus_types.DropCollectionParams(
            name=name
        )

        ## Check that collection exists
        collections: List[str | None] = self.list_collections()
        if name not in collections:
            logger.info(f'❌ Cannot find collection `{name}`.')
            return

        ## Start Drop collection
        logger.info(f'⚙️ Dropping collection `{name}`')
        try:
            # Drop collection
            self.client.drop_collection(collection_name=name)
            # Make sure collection was dropped
            collections = self.list_collections()
            if name not in collections:
                logger.info(f'✅ Dropped collection `{name}`')
            else:
                error_message = f"The collection `name` is still in the collections list."
                logger.error(error_message)
                raise ValueError(error_message)
        except Exception as e:
            logger.error(f'❌ Problem dropping collection: `{str(e)}`')
            raise  


    ## Insert data into a collection
    def insert(
        self, 
        name: str = collection_name, 
        data: List[dict] = data_ex
    ) -> dict | None:
        """
        Insert the given data into the given collection.

        For example, to insert data into a given collection:
        ```python
        # Initialize the client
        uri = 'http://localhost:19530'
        client = MilvusClientInit(uri=uri)

        # Create a collection
        name = 'collection_ex'
        client.create_collection(name=name)

        # Insert some data
        data = [
            {'text': 'information retrieval is a field of study.'},
            {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
            {'text': 'data mining and information retrieval overlap in research.'},
            {'text': 'the rest of the lyrics go,'},
            {'text': 'Last night I dreamed about'}
        ]
        client.insert(name=name, data=data)
        ```

        Args
        ------------
            name: str
                Name of the collection to insert the data into.
            data: List[dict]
                The data to insert into the collection.

        Returns
        ------------
            dict: 
                A dictionary of the data added to the collection.
            
        Raises
        ------------
            Exception: 
                If adding the data fails, error is logged and raised.
        """
        ## Validate all argument types
        milvus_types.InsertParams(
            name=name,
            data=data
        )

        ## Check that collection exists
        collections = self.list_collections()
        if name not in collections:
            logger.info(f'❌ Cannot find collection `{name}`.')
            return None

        ## Start Insert
        logger.info(f'⚙️ Inserting data into `{name}`')
        try:
            # Insert data into the collection
            results: dict = self.client.insert(
                collection_name=name,
                data=data
            )
            # Wait for the database to update
            time.sleep(0.5)
            logger.info(f'✅ Inserted data')

            ## Validate results types
            milvus_types.InsertResults(
                results=results
            )

            ## Return results
            return results
        except Exception as e:
            logger.error(f'❌ Problem inserting data: {str(e)}')
            raise


    ## Delete data from a collection
    def delete(
        self, 
        ids: List[str],
        name: str = collection_name, 
    ) -> None:
        """
        Delete the given data from the given collection.

        For example, to delete data from a given collection:
        ```python
        # Initialize the client
        uri = 'http://localhost:19530'
        client = MilvusClientInit(uri=uri)
        client.delete(ids=['data-id'], name=name)
        ```

        Args
        ------------
            data: List[str]
                A list of ids to delete from the collection.
            name: str
                Name of the collection to upsert the data into.
            
        Raises
        ------------
            Exception: 
                If deleting the data fails, error is logged and raised.
        """
        ## Validate all argument types
        milvus_types.DeleteParams(
            name=name,
            ids=ids
        )

        ## Check that collection exists
        collections = self.list_collections()
        if name not in collections:
            logger.info(f'❌ Cannot find collection `{name}`.')
            return None

        ## Start Delete
        logger.info(f'⚙️ Deleting data from `{name}`')
        try:
            # Delete data into the collection
            results: dict = self.client.delete(
                collection_name=name,
                ids=ids
            )
            # Wait for the database to update
            time.sleep(0.5)
            logger.info(f'✅ Deleted data')
        except Exception as e:
            logger.error(f'❌ Problem deleting data: {str(e)}')
            raise


    ## Perform a full text search on a collection
    def full_text_search(
        self, 
        name: str = collection_name, 
        query_list: List[str] = query_list, 
        limit: int = lim_results
    ) -> SearchResult | None:
        """
        Perform a full text search on the given collection for the queries in the given query list. 
        Get a maximum number of results given by the result limit.

        For example to search a collection for a given list of queries:
        ```python
        # Initialize the client
        uri = 'http://localhost:19530'
        client = MilvusClientInit(uri=uri)

        # Create a collection
        name = 'collection_ex'
        client.create_collection(name=name)

        # Insert some data
        data = [
            {'text': 'information retrieval is a field of study.'},
            {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
            {'text': 'data mining and information retrieval overlap in research.'},
            {'text': 'the rest of the lyrics go,'},
            {'text': 'Last night I dreamed about'}
        ]
        client.insert(name=name, data=data)

        # Query the data
        query_list: List[str] = ['whats the focus of information retrieval?']
        lim_results: int = 3
        client.full_text_search(name=name, query_list=query_list, limit=limit)
        ```

        Args
        ------------
            name: str
                Name of the collection to query.
            query_list: List[str]
                List of queries to perform.
            limit: int
                Maximum number of results to obtain.

        Returns
        ------------
            List[dict]: 
                A list of the search results for the given queries.
            
        Raises
        ------------
            Exception: 
                If performing the search fails, error is logged and raised.
        """
        ## Validate all argument types
        milvus_types.FullTextSearchParams(
            name=name,
            query_list=query_list,
            limit=limit
        )

        ## Check that collection exists
        collections = self.list_collections()
        if name not in collections:
            logger.info(f'❌ Cannot find collection `{name}`.')
            return None

        ## Start Full Text Search
        logger.info(f'⚙️ Performing search on `{name}` for queries: \n {pprint.pformat(query_list, indent=0, width=500)} \n')
        try:
            ## Define search params
            # Only full-text so focus on sparse vectors built from text
            anns_field: str = 'sparse'
            output_fields: List[str] = ['text']

            ## Add extra search params
            # Controls trade-off between speed and accuracy in ANN searches
            # Drop some percentage of results before searching
            search_params: dict = {
                'params': {'drop_ratio_search': 0.2},
            }
            with with_spinner(description=f"🔎 Getting results..."):
                ## Get the search results
                results: List[dict] = self.client.search(
                    collection_name=name, 
                    data=query_list,
                    anns_field=anns_field,
                    output_fields=output_fields,
                    limit=limit,
                    search_params=search_params
                )
            
            ## Validate results types
            milvus_types.FullTextSearchResults(
                results=results
            )

            ## Return results
            for i, result in enumerate(results):
                for j, hit in enumerate(result):
                    logger.info(f'📝 Result {i}, {j}: \n {pprint.pformat(hit, indent=0, width=500)} \n')
            return results
        except Exception as e:
            logger.error(f'❌ Problem performing search: {str(e)}')
            raise