### latency_test
## Executes a simple test of the latency of an Milvus server in Docker

import time
from pyfiles.milvus_utils import MilvusClientInit
from pyfiles.logger import logger
from scripts.generate_dataset import data

logger.info(f'⚙️ Starting latency test in `./scripts/latency_test.py`')

def measure_create_collection_latency(client, name):
    start = time.perf_counter()
    response = client.create_collection(name=name)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Latency: {elapsed_ms:.1f} ms")
    return elapsed_ms

def measure_insert_latency(client, name, data=data['dataset']['data']):
    start = time.perf_counter()
    response = client.insert(name=name, data=data)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Latency: {elapsed_ms:.1f} ms")
    return elapsed_ms

def measure_full_text_search_latency(client, name, query=data['queries']['data']):
    start = time.perf_counter()
    response = client.full_text_search(name=name, query_list=query, limit=5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Latency: {elapsed_ms:.1f} ms")
    return elapsed_ms

def run_test(client, n_tests, method_name):
    latency_sum = 0
    for i in range(n_tests):
        name = f'_test_collection_{i}'
        if method_name=='create_collection':
            latency = measure_create_collection_latency(client, name=name)
            latency_sum += latency
            logger.info(f"Test {i}")
        
        if method_name=='insert':
            latency = measure_insert_latency(client, name=name)
            latency_sum += latency
            logger.info(f"Test {i}")

        if method_name=='full_text_search':
            latency = measure_full_text_search_latency(client, name=name)
            latency_sum += latency
            logger.info(f"Test {i}")
            client.drop_collection(name=name)

    latency_avg = latency_sum/n_tests
    logger.info(f"Latency average for {method_name}: {latency_avg:.1f}")


n_tests = 10
## Initialize Milvus client
# Defaults to host on url 'http://localhost:19530'
client: MilvusClientInit = MilvusClientInit()
# Warmup client
client.list_collections()
# Test create collection latency
run_test(client, n_tests, 'create_collection')
# Test insert data latency
run_test(client, n_tests, 'insert')
# Test full text search latency
run_test(client, n_tests, 'full_text_search')

logger.info(f'✅ Finished latency test in `./scripts/latency_test.py` \n\n')