#!/usr/bin/env python3
"""
Comprehensive RAG Demo
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
from langchain_text_splitters import RecursiveCharacterTextSplitter


def download_data():
    """Download K8s examples and documentation"""
    examples_dir = Path("./k8s-data/examples")
    docs_dir = Path("./k8s-data/docs")

    # Download examples
    if not examples_dir.exists():
        print("Downloading K8s examples...")
        Path("./k8s-data").mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1",
            "https://github.com/kubernetes/examples.git", str(examples_dir)],
            capture_output=True)

    # Download official K8s docs
    if not docs_dir.exists():
        print("Downloading K8s documentation...")
        docs_dir.mkdir(exist_ok=True)
        docs = [
            ("kubectl-cheatsheet.md", "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/reference/kubectl/cheatsheet.md"),
            ("pods.md", "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/pods/_index.md"),
            ("deployments.md", "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/controllers/deployment.md"),
            ("services.md", "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/services-networking/service.md"),
            ("debugging-pods.md", "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/tasks/debug/debug-application/debug-running-pod.md"),
        ]
        for filename, url in docs:
            subprocess.run(["curl", "-s", "-o", str(docs_dir / filename), url], capture_output=True)

    print("Data ready!\n")


def load_documents():
    """Load YAML examples and Markdown documentation"""
    documents = []

    # Load YAML files (excluding archived)
    yaml_loader = DirectoryLoader("./k8s-data/examples/", glob="**/*.yaml",
        loader_cls=TextLoader, silent_errors=True,
        loader_kwargs={"autodetect_encoding": True})
    yaml_docs = [d for d in yaml_loader.load() if "_archived" not in d.metadata.get("source", "")]
    documents.extend(yaml_docs)

    # Load documentation
    if Path("./k8s-data/docs/").exists():
        md_loader = DirectoryLoader("./k8s-data/docs/", glob="**/*.md",
            loader_cls=TextLoader, silent_errors=True,
            loader_kwargs={"autodetect_encoding": True})
        documents.extend(md_loader.load())

    return documents


def chunk_documents(documents):
    """Split large documents into smaller chunks for better retrieval"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


def format_source(source_path):
    """Convert local path to GitHub URL"""
    if "k8s-data/examples/" in source_path:
        path = source_path.split("k8s-data/examples/")[-1]
        return f"https://github.com/kubernetes/examples/blob/master/{path}"
    elif "k8s-data/docs/" in source_path:
        return f"K8s Docs: {Path(source_path).name}"
    return source_path


def main():
    print("\n" + "=" * 50)
    print("  Comprehensive RAG Demo - K8s Knowledge Base")
    print("=" * 50 + "\n")

    # Step 1: Get data
    download_data()

    # Step 2: Load and chunk documents
    print("Loading documents...")
    documents = load_documents()
    print(f"  Found {len(documents)} documents")

    print("Chunking for better retrieval...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks\n")

    # Step 3: Build vector store
    print("Building search index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("  Index ready!\n")

    # Step 4: Connect LLM
    print("Connecting to Ollama (llama3.2)...")
    llm = OllamaLLM(model="llama3.2")
    print("  Connected!\n")

    # Step 5: Create smart RAG chain
    prompt = PromptTemplate.from_template("""You are a Kubernetes expert. Answer based ONLY on the context below.
If the question is unrelated to Kubernetes or the context doesn't help, say: "I don't have information about that."

Context:
{context}

Question: {question}

Provide a helpful, detailed answer with examples when relevant:""")

    chain = (
        {"context": retriever | (lambda docs: "\n\n---\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    # Step 6: Interactive Q&A with sources
    print("=" * 50)
    print("Ready! Ask anything about Kubernetes")
    print("Try: 'How do I get logs from a pod?'")
    print("     'What is a deployment?'")
    print("     'Show me a service example'")
    print("=" * 50 + "\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not query:
            continue

        print("\nSearching knowledge base...")

        # Get answer
        answer = chain.invoke(query)
        print(f"\nAnswer:\n{answer}\n")

        # Show sources (only if relevant answer)
        if "don't have information" not in answer.lower():
            docs = retriever.invoke(query)
            print("Sources:")
            seen = set()
            for doc in docs:
                src = format_source(doc.metadata.get("source", "unknown"))
                if src not in seen:
                    print(f"  • {src}")
                    seen.add(src)

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
