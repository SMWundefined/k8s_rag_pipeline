#!/usr/bin/env python3
"""
GenAI 101: Simple RAG Demo for K8s Docs
"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


def main():
    print("\n=== RAG Demo: Search K8s Docs with AI ===\n")

    # STEP 1: Load documents
    print("1. Loading K8s documentation...")
    loader = DirectoryLoader(
        "./k8s-data/examples/",
        glob="**/*.yaml",
        loader_cls=TextLoader,
        silent_errors=True,
        loader_kwargs={"autodetect_encoding": True},
    )
    documents = loader.load()
    print(f"   Loaded {len(documents)} files\n")

    # STEP 2: Create embeddings (convert text to vectors)
    print("2. Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # STEP 3: Store in vector database
    print("3. Building vector database...")
    vectorstore = Chroma.from_documents(
        documents, embeddings, persist_directory="./chroma_db"
    )

    # STEP 4: Connect to LLM
    print("4. Connecting to Ollama...\n")
    llm = OllamaLLM(model="llama3.2")

    # STEP 5: Create RAG chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        """Use this context to answer the question:

Context: {context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_sources(docs):
        return [doc.metadata.get("source", "unknown") for doc in docs]

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # STEP 6: Interactive Q&A
    print("Ready! Ask questions about Kubernetes (type 'quit' to exit)\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            print("\nSearching...")
            docs = retriever.invoke(query)
            answer = rag_chain.invoke(query)
            print(f"\nAnswer: {answer}\n")
            print("Sources:")
            for doc in docs:
                print(f"  - {doc.metadata.get('source', 'unknown')}")
            print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
