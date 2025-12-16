#!/usr/bin/env python3
"""
GenAI 101: Simple RAG Demo (Fast Version)
- Loads only YAML files
- Minimal setup, quick results
"""

import shutil
import subprocess
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


def main():
    print("\n=== RAG Demo (Fast Version) ===\n")

    # Download K8s examples if needed
    examples_dir = Path("./k8s-data/examples")
    if not examples_dir.exists():
        print("Downloading K8s examples...")
        Path("./k8s-data").mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1",
            "https://github.com/kubernetes/examples.git",
            str(examples_dir)], capture_output=True)
        print("Done!\n")

    # Load YAML files only (faster), excluding archived content
    print("1. Loading YAML files...")
    loader = DirectoryLoader("./k8s-data/examples/", glob="**/*.yaml",
        loader_cls=TextLoader, silent_errors=True,
        loader_kwargs={"autodetect_encoding": True})
    documents = [doc for doc in loader.load() if '_archived' not in doc.metadata.get('source', '')]

    # Also load K8s documentation
    docs_dir = Path("./k8s-data/docs")
    if docs_dir.exists():
        loader_docs = DirectoryLoader("./k8s-data/docs/", glob="**/*.md",
            loader_cls=TextLoader, silent_errors=True,
            loader_kwargs={"autodetect_encoding": True})
        documents.extend(loader_docs.load())

    print(f"   Loaded {len(documents)} files\n")

    # Create embeddings
    print("2. Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build vector database
    print("3. Building vector database...")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db_simple")

    # Connect to LLM
    print("4. Connecting to Ollama...\n")
    llm = OllamaLLM(model="llama3.2")

    # Create RAG chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate.from_template(
        "Context: {context}\n\nQuestion: {question}\n\nProvide a brief, concise answer:")

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    # Interactive Q&A
    print("Ready! Ask questions (type 'quit' to exit)\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            print("\nSearching...")
            docs = retriever.invoke(query)
            print(f"\nAnswer: {rag_chain.invoke(query)}\n")
            print("Sources:")
            for doc in docs:
                print(f"  - {doc.metadata.get('source', 'unknown')}")
            print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
