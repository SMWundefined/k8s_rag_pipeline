#!/usr/bin/env python3
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def main():
    print("\n" + "="*60)
    print(" GenAI 101: RAG for K8s Docs (Free Version)")
    print("="*60 + "\n")
    
# STEP 1: Load K8s Sample Files
    print(" Step 1: Loading K8s configs...")
    
    loader = DirectoryLoader(
        './sample-data/',
        glob="**/*.{yaml,yml,md}",
        loader_cls=TextLoader,
        show_progress=False
    )
    
    documents = loader.load()
    print(f"    Loaded {len(documents)} files\n")
    
    if not documents:
        print(" No files found in ./sample-data/")
        print("   Make sure you have YAML or Markdown files there!")
        return
    
# STEP 2: Create Embeddings (FREE!)
    print(" Step 2: Creating embeddings...")
    print("   (Converting text to 'coordinates' that capture meaning)")
    print("   Using HuggingFace - 100% free!\n")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
# STEP 3: Store in Vector Database
    print(" Step 3: Building vector database...")
    print("   (Creating the 'map' of concepts)\n")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
# STEP 4: Connect to Local LLM (FREE!)
    print(" Step 4: Connecting to Ollama...")
    print("   (Local LLM - no API calls!)\n")
    
    try:
        llm = Ollama(model="llama3.2")
    except Exception as e:
        print(" Error connecting to Ollama.")
        print("   Make sure Ollama is running:")
        print("   1. Install: https://ollama.ai")
        print("   2. Run: ollama pull llama3.2")
        print("   3. Ollama runs automatically after install")
        return
    
# STEP 5: Create RAG System
    print(" Step 5: Creating RAG system...\n")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    print("="*60)
    print(" Ready! Ask questions about Kubernetes")
    print("="*60)
    print("\n  Try asking:")
    print("   • 'Show me a deployment example'")
    print("   • 'How do I configure persistent storage?'")
    print("   • 'What's a service?'")
    print("\n Type 'quit' to exit\n")
    
# STEP 6: Interactive Q&A
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for learning about RAG!")
            break
        
        if not query:
            continue
        
        try:
            print("\n Searching...")
            result = qa_chain.invoke({"query": query})
            print(f"\n Answer:\n{result['result']}\n")
            print("-" * 60 + "\n")
            
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    # Check if sample data exists
    if not os.path.exists('./sample-data'):
        print("Error: ./sample-data/ directory not found!")
        print("Make sure you're in the workshop directory.")
        exit(1)
    
    main()
