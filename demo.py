#!/usr/bin/env python3
"""
GenAI 101: RAG for K8s Docs (Free Version)
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def clone_k8s_docs():
    """Clone Kubernetes examples if not already present"""
    data_dir = Path("./k8s-data")
    examples_dir = data_dir / "examples"

    if examples_dir.exists() and any(examples_dir.rglob("*.yaml")):
        print("K8s examples already downloaded")
        return True

    print("Downloading Kubernetes examples (one-time setup)...")
    print("This will take 1-2 minutes...")

    data_dir.mkdir(exist_ok=True)

    # Remove any failed partial download
    if examples_dir.exists():
        shutil.rmtree(examples_dir)

    try:
        # Simple shallow clone - most reliable
        subprocess.run([
            "git", "clone",
            "--depth", "1",
            "https://github.com/kubernetes/examples.git",
            str(examples_dir)
        ], check=True, capture_output=True)

        print("Download complete\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error downloading K8s examples: {e}")
        print("Falling back to minimal documentation...")
        return False

def main():
    print("\n" + "="*60)
    print("GenAI 101: RAG for K8s Docs (Free Version)")
    print("="*60 + "\n")

    # STEP 0: Download K8s Documentation
    if not clone_k8s_docs():
        print("ERROR: Could not download K8s documentation")
        print("Please check your internet connection and git installation")
        return

    # STEP 1: Load K8s Documentation
    print("Step 1: Loading K8s documentation...")

    # Try loading YAML files
    try:
        loader = DirectoryLoader(
            './k8s-data/examples/',
            glob="**/*.yaml",
            loader_cls=TextLoader,
            show_progress=False,
            silent_errors=True,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents = loader.load()

        # Also try .yml files
        loader_yml = DirectoryLoader(
            './k8s-data/examples/',
            glob="**/*.yml",
            loader_cls=TextLoader,
            show_progress=False,
            silent_errors=True,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents.extend(loader_yml.load())

        # Also try .md files
        loader_md = DirectoryLoader(
            './k8s-data/examples/',
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=False,
            silent_errors=True,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents.extend(loader_md.load())

    except Exception as e:
        print(f"ERROR loading files: {e}")
        documents = []

    print(f"Loaded {len(documents)} K8s files\n")

    if not documents:
        print("ERROR: No files found in k8s-data/examples/")
        print("Debug info:")
        from pathlib import Path
        p = Path("./k8s-data/examples/")
        yaml_count = len(list(p.rglob("*.yaml")))
        print(f"  YAML files found on disk: {yaml_count}")
        print("\nTry:")
        print("  1. Check encoding: python debug.py")
        print("  2. Or delete k8s-data/ and run again")
        return

    # STEP 2: Create Embeddings (FREE)
    print("Step 2: Creating embeddings...")
    print("Converting text to vectors that capture meaning")
    print("Using HuggingFace - 100% free\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # STEP 3: Store in Vector Database
    print("Step 3: Building vector database...")
    print("Creating searchable index of K8s documentation\n")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # STEP 4: Connect to Local LLM (FREE)
    print("Step 4: Connecting to Ollama...")
    print("Using local LLM - no API calls\n")

    try:
        llm = OllamaLLM(model="llama3.2")
    except Exception as e:
        print("ERROR: Could not connect to Ollama")
        print("Make sure Ollama is running:")
        print("  1. Install: https://ollama.ai")
        print("  2. Run: ollama pull llama3.2")
        print("  3. Ollama runs automatically after install")
        return

    # STEP 5: Create RAG System
    print("Step 5: Creating RAG system...\n")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = PromptTemplate.from_template(
        """Use the following context to answer the question. If you don't know the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("="*60)
    print("Ready! Ask questions about Kubernetes")
    print("="*60)
    print("\nTry asking:")
    print("  - 'Show me a deployment example'")
    print("  - 'How do I configure persistent storage?'")
    print("  - 'What's a service?'")
    print("\nType 'quit' to exit\n")

    # STEP 6: Interactive Q&A
    while True:
        query = input("Your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for learning about RAG!")
            break

        if not query:
            continue

        try:
            print("\nSearching...")
            result = rag_chain.invoke(query)
            print(f"\nAnswer:\n{result}\n")
            print("-" * 60 + "\n")

        except Exception as e:
            print(f"\nERROR: {str(e)}\n")

if __name__ == "__main__":
    # Check if running in correct directory
    if not os.path.exists('./demo.py'):
        print("ERROR: Please run from the workshop directory")
        sys.exit(1)

    main()
