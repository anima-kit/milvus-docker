### milvus_types
## Type validation for methods in milvus_utils

## Imports
# Third-party modules
from pydantic import (
    BaseModel, 
    field_validator
)

from pymilvus.milvus_client.index import IndexParams # type: ignore
from pymilvus import ( # type: ignore
    MilvusClient,       
    CollectionSchema,   
    Function,           
    FunctionType,       
    DataType            
) 
from pymilvus.client.search_result import ( # type: ignore
    SearchResult, 
    HybridHits, 
    Hit
) 

from typing import (
    List, 
    Any
)

# Internal modules
from pyfiles.logger import (
    logger, 
    with_spinner
)


class InitClientParams(BaseModel):
    """
    Parameters required to initialize the `MilvusClientInit`.

    Attributes
    ------------
        uri: str
            The URI of the Milvus server, e.g., 'http://localhost:19530'.
    """
    uri: str

    @field_validator('uri')
    @classmethod
    def validate_uri(cls, v: str) -> str:
        if not isinstance(v, str):
            error_message = f"The `uri` argument should be a string, instead got `{type(v)}`."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class InitClientResults(BaseModel):
    """
    Results from initiliazing the `MilvusClientInit`.

    Attributes
    ------------
        results: MilvusClient
            The client attribute of `MilvusClientInit`.
    """
    results: Any

    @field_validator('results')
    @classmethod
    def validate_results(cls, v: MilvusClient) -> MilvusClient:
        if not isinstance(v, MilvusClient):
            error_message = f"The results from the `_init_client` method should be a `MilvusClient`, instead got `{type(v)}`."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class CreateFieldParams(BaseModel):
    """
    Parameters required for the `MilvusClientInit._create_field` method.

    Attributes
    ------------
        collection_schema: CollectionSchema
            The schema for a given collection (created with MilvusClient.create_schema).
        params: dict
            The field parameters to add to the schema (using schema.add_field).
    """
    collection_schema: Any
    params: dict

    @field_validator('collection_schema')
    @classmethod
    def validate_collection_schema(cls, v: CollectionSchema) -> CollectionSchema:
        if not isinstance(v, CollectionSchema):
            error_message = f"The argument `schema` should be a `CollectionSchema`, instead got `{type(v)}`."
            logger.error(error_message)
            raise TypeError(error_message)
        return v

    @field_validator('params')
    @classmethod
    def validate_params(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            error_message = f"The argument `params` should be a dictionary, instead got `{type(v)}`."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class CreateIndexParams(BaseModel):
    """
    Parameters required for the `MilvusClientInit._create_index` method.

    Attributes
    ------------
        index_params: IndexParams
            The index parameters for a given collection (created with MilvusClient.prepare_index_params).
        params: dict
            The indices to add to the index_parameters (using index_params.add_index).
    """
    index_params: Any
    params: dict

    @field_validator('index_params')
    @classmethod
    def validate_index_params(cls, v: IndexParams) -> IndexParams:
        if not isinstance(v, IndexParams):
            error_message = f"The `index_params` argument should be an `IndexParams`, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        return v

    @field_validator('params')
    @classmethod
    def validate_params(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            error_message = f"The `params` argument should be an dictionary, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class ListCollectionsResults(BaseModel):
    """
    Results from listing all collections for a client by invoking `MilvusClientInit.list_collections`.

    Attributes
    ------------
        results: List[str | None]
            The resulting list of all collections.
    """
    results: List[str | None]

    @field_validator('results')
    @classmethod
    def validate_results(cls, v: list) -> list:
        if not isinstance(v, list):
            error_message = f"The results from the `list_collections` method should be a List[str | None], instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        if not all(isinstance(item, str) for item in v):
            error_message = f"Each item in the collections list should be a string."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class CreateCollectionParams(BaseModel):
    """
    Parameters required for the `MilvusClientInit.create_collection` method.

    Attributes
    ------------
        name: str
            The name of the collection to create.
        field_params_list: List[dict]
            The list of field parameters to add to the schema.
        func_list: Function
            The list of functions to add to the schema.
        index_params_list: List[dict]
            The list of indices to add to the index parameters
    """
    name: str
    field_params_list: List[dict]
    func_list: List[Any]
    index_params_list: List[dict]

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not isinstance(v, str):
            error_message = f"The `name` argument should be a string, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        return v

    @field_validator('field_params_list')
    @classmethod
    def validate_field_params_list(cls, v: list) -> list:
        if not isinstance(v, list):
            error_message = f"The `field_params_list` argument should be a List[dict], instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        if not all(isinstance(item, dict) for item in v):
            error_message = f"Each item in `field_params_list` should be a dictionary."
            logger.error(error_message)
            raise TypeError(error_message)
        return v

    @field_validator('func_list')
    @classmethod
    def validate_func_list(cls, v: list) -> list:
        if not isinstance(v, list):
            error_message = f"The `func_list` argument should be a `List[Function]`, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        if not all(isinstance(item, Function) for item in v):
            error_message = f"Each item in `func_list` should be a Function."
            logger.error(error_message)
            raise TypeError(error_message)
        return v

    @field_validator('index_params_list')
    @classmethod
    def validate_index_params_list(cls, v: list) -> list:
        if not isinstance(v, list):
            error_message = f"The `index_params_list` argument should be a List[dict], instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        if not all(isinstance(item, dict) for item in v):
            error_message = f"Each item in `index_params_list` should be a dictionary."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class DropCollectionParams(BaseModel):
    """
    Parameters required for the `MilvusClientInit.drop_collection` method.

    Attributes
    ------------
        name: str
            The name of the collection to drop.
    """
    name: str

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not isinstance(v, str):
            error_message = f"The `name` argument should be a string, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class InsertParams(BaseModel):
    """
    Parameters required for the `MilvusClientInit.insert` method.

    Attributes
    ------------
        name: str
            The name of the collection to create.
        data: List[dict]
            The list of data to add to the collection.
    """
    name: str
    data: List[dict]

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not isinstance(v, str):
            error_message = f"The `name` argument should be a string, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        return v

    @field_validator('data')
    @classmethod
    def validate_data(cls, v: list) -> list:
        if not isinstance(v, list):
            error_message = f"The `data` argument should be a `List[dict]`, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        if not all(isinstance(item, dict) for item in v):
            error_message = f"Each item in `data` should be a dictionary."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class InsertResults(BaseModel):
    """
    Results from inserting data into a collection by invoking `MilvusClientInit.insert`.

    Attributes
    ------------
        results: dict
            The resulting dictionary of the inserted data.
    """
    results: dict

    @field_validator('results')
    @classmethod
    def validate_results(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            error_message = f"Results from the `insert` method should be a dictionary, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class DeleteParams(BaseModel):
    """
    Parameters required for the `MilvusClientInit.delete` method.

    Attributes
    ------------
        name: str
            The name of the collection to create.
        ids: List[str]
            The list of IDs to delete from the collection.
    """
    name: str
    ids: List[str]

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not isinstance(v, str):
            error_message = f"The `name` argument should be a string, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        return v

    @field_validator('ids')
    @classmethod
    def validate_data(cls, v: list) -> list:
        if not isinstance(v, list):
            error_message = f"The `data` argument should be a `List[str]`, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        if not all(isinstance(item, str) for item in v):
            error_message = f"Each item in `data` should be a string."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class FullTextSearchParams(BaseModel):
    """
    Parameters required for the `MilvusClientInit.full_text_search` method.

    Attributes
    ------------
        name: str
            The name of the collection to create.
        query_list: List[str]
            The list of queries for which to do a search.
        limit: int
            The maximum number of results to obtain.
    """
    name: str
    query_list: List[str]
    limit: int

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not isinstance(v, str):
            error_message = f"The `name` argument should be a string, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        return v

    @field_validator('query_list')
    @classmethod
    def validate_query_list(cls, v: list) -> list:
        if not isinstance(v, list):
            error_message = f"The `query_list` argument should be a `List[str]`, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        if not all(isinstance(item, str) for item in v):
            error_message = f"Each item in `query_list` should be a string."
            logger.error(error_message)
            raise TypeError(error_message)
        return v

    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v: int) -> int:
        if not isinstance(v, int):
            error_message = f"The `limit` argument should be an integer, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        return v


class FullTextSearchResults(BaseModel):
    """
    Results from doing a full text search by invoking `MilvusClientInit.full_text_search`.

    Attributes
    ------------
        results: SearchResult
            The results from a full text search.
    """
    results: Any

    @field_validator('results')
    def validate_results(cls, v):
        if not isinstance(v, SearchResult):
            error_message = f"Full text search results should be a `SearchResult`, instead got {type(v)}."
            logger.error(error_message)
            raise TypeError(error_message)
        if not all(isinstance(item, HybridHits) for item in v):
            error_message = "Each item in the list of `results` should be a `HybridHits`."
            logger.error(error_message)
            raise TypeError(error_message)
        for result in v:
            if not all(isinstance(item, Hit) for item in result):
                error_message = "Each item in the list of `result` should be a `Hit`."
                logger.error(error_message)
                raise TypeError(error_message)
        return v