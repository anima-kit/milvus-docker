### milvus_test
## Executes a simple test to check that a Milvus server in Docker can be properly invoked in a Python environment 
## The script runs as follows:
#   - Initialize MilvusClientInit
#   - Manage example collection creation
#   - Insert example data
#   - Do a full text search for example query
#   - Drop collection to cleanup

## Imports
# Third-party modules
from typing import List

# Internal modules
from pyfiles.milvus_utils import MilvusClientInit, collection_name
from pyfiles.logger import logger

logger.info(f'⚙️ Starting Milvus test in `./milvus_test.py`')

## Initialize Milvus client
# Defaults to host on URI | 'http://localhost:19530'
client = MilvusClientInit()

## Drop collection if it exists, then create collection
collection_name: str = collection_name
collections: List[str | None] = client.list_collections()
if collection_name in collections:
    logger.info(f'⚠️ `{collection_name}` already exists, dropping it')
    client.drop_collection(collection_name)
client.create_collection()

## Insert data
# Defaults to inserting a list of dictionaries with text (see `data_ex` of `milvus_types.py) into the `collection_ex` collection
client.insert()

## Get search results
# Defaults to query the `collection_ex` collection
# Searches for a default query list given by `query_list` in `milvus_utils.py`
# Defaults to a maximum of 3 results
client.full_text_search()

## Drop collection to clean up at end
client.drop_collection(collection_name)

logger.info(f'✅ Finished Milvus test in `./milvus_test.py` \n\n')