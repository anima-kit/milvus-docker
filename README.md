![GitHub Workflow Status](https://github.com/anima-kit/milvus-docker/actions/workflows/ci.yml/badge.svg?branch=main) [![codecov](https://codecov.io/gh/anima-kit/milvus-docker/graph/badge.svg)](https://codecov.io/gh/anima-kit/milvus-docker)

This project was a way for me to learn how to create and utilize containers in Docker for building AI systems. There's no roadmap for future features, and won't be maintained.

You can [checkout tutorials here][tutorials] for concepts pertaining to building AI systems.

# <img src="https://anima-kit.github.io/milvus-docker/assets/milvus.svg" alt="Milvus" style="width: 32px; height: 32px; vertical-align: middle;"> <img src="https://anima-kit.github.io/milvus-docker/assets/docker.svg" alt="Docker" style="width: 32px; height: 32px; vertical-align: middle;"> <img src="https://anima-kit.github.io/milvus-docker/assets/python.svg" alt="Python" style="width: 32px; height: 32px; vertical-align: middle;">  Milvus Docker with Python

![image](https://anima-kit.github.io/milvus-docker/assets/milvus-docker-python.png)

<p align="center">
  <img src="https://anima-kit.github.io/milvus-docker/assets/milvus-docker-intro.gif" alt="animated" />
</p>

## 🔖 About This Project 

> TL;DR
Learn how to use a vector database on your local machine to store and search your data. Then, you can use this setup as a tool to give to [locally run AI agents][tutorials] 🤖.

This repo demonstrates how to set up a [Milvus][milvus] server in [Docker][docker] on your local machine and use it within a [Python][python] environment to store and query your data. The Milvus server utilizes a [MinIO][minio] server for data storage and an [etcd][etcd] server for storage and coordination. It serves as part of the foundation for building AI agents by giving them the ability to obtain relevant information about custom data.

The Docker setup for this repo is based on the [official Docker setup from Milvus][milvus-docker] and the Python methods use the [PyMilvus] library. See the [license section][license-section] for more details.

This project is part of my broader goal to create tutorials and resources for building agents with [LangChain][langchain] and [LangGraph][langgraph]. For more details about how to use this repo and other easily digestible modules to build agents, [check it out here][animakit].

Now, let's get building!

## 🏁 Getting Started 

1.  Make sure [Docker][docker] is installed and running.

1.  Clone the repo, head there, then create a Python environment:

    ```bash
    git clone https://github.com/anima-kit/milvus-docker.git
    cd milvus-docker
    python -m venv venv
    ```

    <a id="gs-activate"></a>

1.  Activate the Python environment:

    ```bash
    venv/Scripts/activate
    ```

1.  Install the necessary Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

    <a id="gs-start"></a>

1.  Build and start all the Docker containers:

    ```bash
    docker compose up -d
    ```

1.  Head to [http://127.0.0.1:9091/webui/][milvus-webui] to check out some useful Milvus client information.

1.  Run the test script to ensure the Milvus server can be reached through the [PyMilvus library][pymilvus]:

    ```bash
    python -m scripts.milvus_test
    ```

    <a id="gs-stop"></a>

1.  When you're done, stop the Docker containers and cleanup with:

    ```bash
    docker compose down
    ```

## 📝 Example Use Cases 

After setting everything up, you can now add and search your own data through the provided Python methods.

The main class to interact with the vector database is the `MilvusClientInit` class which is built on the [PyMilvus][pymilvus] library. Once this class is initialized, you can manage collections and collection data, as well as search a collection for given queries. 

This repo demonstrates how to do a *full-text search*. [In a future tutorial][doc-agent-tutorial], I'll show how to do a hybrid search, which is a combination of a full-text search (sparse vectors) and a dense vector search to capture semantic meaning.

For example, to manage your data and perform a full-text search through a custom script, follow these steps:

1.  Do [step 3][step-activate] and [step 5][step-start] of the `🏁 Getting Started` section to activate the Python environment and run the Milvus server.

    <a id="rs-create"></a>

1.  Create a script named `my-data-search-ex.py` with the following:

    ```python
    # Import MilvusClientInit class
    from pyfiles.milvus_utils import MilvusClientInit

    # Initialize client
    client = MilvusClientInit()

    # Create collection
    collection_name = 'my_collection'
    client.create_collection(collection_name)

    # Create data and insert into collection
    my_data = [
        {'text': 'grocery list: bananas, bread, choco'},
        {'text': 'grocery: green beans'},
        {'text': 'todo list: start chatbot tutorial, network with community'},
        {'text': 'study list: langchain v1, lucid dreaming and asc'},
        {'text': 'study latest gradio implements'},
        {'text': 'My dream last night involved'},
        {'text': 'then I woke up, confused as to if I was still dreaming.'}
    ]
    client.insert(name=collection_name, data=my_data)

    # Define the maximum number of results and the search query
    num_results = 2
    query_list = ['grocery', 'study', 'dream']

    # Get results
    client.full_text_search(
        name=collection_name, 
        query_list=query_list, 
        limit=num_results
    )

    # (Optional) Delete collection when done to clean up
    client.drop_collection(name=collection_name)
    ```

    <a id="rs-run"></a>

1.  Run the script

    ```bash
    python my-data-search-ex.py
    ```

1.  Do [step 8][step-stop] of the `Getting Started 🏁` section to stop the containers and cleanup when you're done.

Milvus also allows for a rich customization of search types and index parameters. For a more detailed discussion of what can be done with this repo and with Milvus in general, [check out the companion tutorial here][milvus-tutorial].

## 📚 Next Steps & Learning Resources 

This project is part of a series on building AI agents. For a deeper dive, [check out my tutorials][tutorials]. Topics include:

- Setting up local servers (like this one) to power the agent
- Example agent workflows (simple chatbots to specialized agents)
- Implementing complex RAG techniques
- Discussing various aspects of AI beyond agents

Want to learn how to expand this setup? [Visit my portfolio][animakit] to explore more tutorials and projects!

## 🏯 Project Structure

```
├── docker-compose.yml      # Docker configurations
├── pyfiles/                # Python source code
│   └── logger.py           # Python logger for tracking progress
│   └── milvus_utils.py     # Python methods to use Milvus server
├── requirements.txt        # Required Python libraries for main app
├── requirements-dev.txt    # Required Python libraries for development
├── scripts/                # Example scripts to use Python methods
├── tests/                  # Testing suite
├── third-party/            # Milvus/PyMilvus licensing
└── validators/             # Validators for Python methods
```

## ⚙️ Tech 

- [Milvus][milvus]: Vector database setup in Docker
- [PyMilvus][pymilvus]: Interacting with Milvus in Python 
- [etcd][etcd]: Data storage and coordination
- [MinIO][minio]: Data storage 
- [Docker][docker]: Setup of all containers

## 🔗 Contributing 

This repo is a work in progress. If you'd like to suggest or add improvements, fix bugs or typos etc., feel free to contribute. Check out the [contributing guidelines][contributing] to get started.

<a id="license-section"></a>

## 📑 License

This repo is licensed under [MIT][license]. However, note that the Docker setup for this repo is based on the [official Docker setup from Milvus][milvus-docker] and the Python methods utilize the [PyMilvus][pymilvus] library. Both [Milvus][milvus] and [PyMilvus][pymilvus] are licensed under Apache 2.0. See the full [Milvus license here][milvus-license] and the full [PyMilvus license here][pymilvus-license]. 

<!-- LINKS -->
[animakit]: https://anima-kit.github.io/
[contributing]: CONTRIBUTING.md
[doc-agent-tutorial]: https://anima-kit.github.io/tutorials/agents/doc-agent/
[docker]: https://www.docker.com/
[etcd]: https://etcd.io/
[langchain]: https://www.langchain.com/
[langgraph]: https://www.langchain.com/langgraph/
[license]: LICENSE
[license-section]: https://github.com/anima-kit/milvus-docker/blob/main/README.md#license-section
[milvus]: https://milvus.io/
[milvus-license]: https://github.com/anima-kit/milvus-docker/blob/main/third-party/milvus-LICENSE
[milvus-docker]: https://github.com/milvus-io/milvus/releases
[milvus-tutorial]: https://anima-kit.github.io/tutorials/servers/milvus/
[milvus-webui]: http://127.0.0.1:9091/webui/
[minio]: https://www.min.io/
[pymilvus]: https://github.com/milvus-io/pymilvus
[pymilvus-license]: https://github.com/anima-kit/milvus-docker/blob/main/third-party/pymilvus-LICENSE
[python]: https://www.python.org/
[step-activate]: https://github.com/anima-kit/milvus-docker/blob/main/README.md#gs-activate
[step-start]: https://github.com/anima-kit/milvus-docker/blob/main/README.md#gs-start
[step-stop]: https://github.com/anima-kit/milvus-docker/blob/main/README.md#gs-stop
[tutorials]: https://anima-kit.github.io/tutorials/