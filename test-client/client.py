from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter
import asyncio

DATA_DIR = "data"
API_URL="http://oas-tailscale:11434"

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.3", request_timeout=360.0, base_url=API_URL)


def load_local_index():
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    return load_index_from_storage(storage_context)

def generate_local_index():
    print("Index not found, creating a new one.")
    all_documents = SimpleDirectoryReader(DATA_DIR).load_data()
    md_documents = [doc for doc in all_documents if doc.metadata.get('file_path', '').endswith('.md')]

    # Add to your existing code
    markdown_splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separator="\n",  # Split at newlines for markdown
        backup_separators=["\n## ", "\n### "]  # Split at headers
    )

    # In your index creation
    index = VectorStoreIndex.from_documents(
        md_documents,
        transformations=[markdown_splitter],
        show_progress=True
    )

    index.storage_context.persist("storage")
    return index

def load_index(): # TODO need a way to detect file changes and update index
    try:
        print("Loading index from storage.")
        index = load_local_index()
    except FileNotFoundError:
        print("Index not found, creating a new one.")
        index = generate_local_index()
    return index


index = load_index()
query_engine = index.as_query_engine()


async def search_documents(query: str) -> str:
    response = await query_engine.aquery(query)
    return str(response)


agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can search through documents to answer questions.""",
)


async def main():
    response = await agent.run(
        "What is my favorite number?"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
