#!/usr/bin/env python3
"""
Simple RAG Model for Kubernetes context
"""

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
    print("\n=== Simple RAG Demo ===\n")

    # Step 1: Get K8s examples
    data_dir = Path("./k8s-data/examples")
    if not data_dir.exists():
        print("Downloading K8s examples...")
        subprocess.run(["git", "clone", "--depth", "1",
            "https://github.com/kubernetes/examples.git", str(data_dir)],
            capture_output=True)

    # Step 2: Load documents
    print("Loading documents...")
    loader = DirectoryLoader(str(data_dir), glob="**/*.yaml",
        loader_cls=TextLoader, silent_errors=True)
    docs = loader.load()
    print(f"Loaded {len(docs)} files")

    # Step 3: Create vector store
    print("Building search index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 4: Connect LLM
    print("Connecting to LLM...")
    llm = OllamaLLM(model="llama3.2")

    # Step 5: Create RAG chain
    prompt = PromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")

    chain = (
        {"context": retriever | (lambda d: "\n".join(x.page_content for x in d)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    # Step 6: Interactive loop
    print("\nReady! Ask about Kubernetes (type 'quit' to exit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ["quit", "exit", "q"]: break
        if q:
            print(f"\nBot: {chain.invoke(q)}\n")

if __name__ == "__main__":
    main()
