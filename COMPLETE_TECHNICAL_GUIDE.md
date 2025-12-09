# RAG System - Complete Technical Overview

## The Process - What Actually Happens

### Phase 1: Setup (One-Time, ~3-4 minutes)

**Step 1: Download Documentation**
```
Action: Git clone kubernetes/examples
What happens: Downloads ~400KB of YAML and Markdown files
Result: 319 files in k8s-data/examples/
```

**Step 2: Load Documents**
```
Action: DirectoryLoader reads all .yaml, .yml, .md files
What happens: Reads file contents into memory
Result: 319 Document objects with content + metadata
```

**Step 3: Create Embeddings**
```
Action: sentence-transformers converts text to vectors
What happens: 
  - Each document becomes a 384-dimensional vector
  - "deployment" → [0.023, -0.145, 0.089, ...]
  - Similar meanings = similar vectors
  - Downloads model first time (~90MB)
Result: 319 embedding vectors
Time: ~30-60 seconds
```

**Step 4: Store in Vector Database**
```
Action: ChromaDB indexes the vectors
What happens:
  - Creates HNSW index for fast similarity search
  - Stores vectors + original text + metadata
  - Saves to disk in ./chroma_db/
Result: Searchable index ready for queries
```

### Phase 2: Query (Every Question, ~2-5 seconds)

**Step 1: User Asks Question**
```
Input: "Show me a deployment with 3 replicas"
```

**Step 2: Convert Question to Embedding**
```
Action: sentence-transformers creates vector for question
What happens: Question becomes 384-dimensional vector
Result: [0.034, -0.156, 0.091, ...]
Time: ~100ms
```

**Step 3: Semantic Search**
```
Action: ChromaDB finds similar vectors
What happens:
  - Computes cosine similarity between query vector and all document vectors
  - Ranks by similarity score
  - Returns top K matches (K=3 by default)
Algorithm: Approximate Nearest Neighbors (ANN)
Time: ~50ms
```

**Step 4: Retrieve Context**
```
Action: Get full text of top 3 matching documents
Result: 3 YAML files or markdown sections with highest relevance
Example matches:
  1. deployment.yaml (score: 0.89)
  2. deployment-guide.md (score: 0.82)
  3. replicaset-example.yaml (score: 0.76)
```

**Step 5: Build Prompt**
```
Action: Combine question + retrieved context
What happens: Creates prompt like:

  Context:
  [deployment.yaml content]
  [deployment-guide.md content]
  [replicaset-example.yaml content]
  
  Question: Show me a deployment with 3 replicas
  
  Answer based on the context above:
```

**Step 6: Generate Answer**
```
Action: Send prompt to Ollama (llama3.2)
What happens:
  - LLM reads context
  - Generates answer based on retrieved docs
  - Returns natural language response
Time: ~2-4 seconds (local inference)
```

**Step 7: Return to User**
```
Output: Answer with YAML example and explanation
```

## Behind the Scenes - Technical Details

### Embeddings - How Meaning Becomes Numbers

**What are embeddings?**
- Vectors (lists of numbers) that capture semantic meaning
- Similar concepts have similar vectors
- 384 dimensions for all-MiniLM-L6-v2 model

**Example:**
```
"deployment"     → [0.023, -0.145, 0.089, ..., 0.156]
"replicaset"     → [0.028, -0.141, 0.092, ..., 0.149]  # Similar!
"coffee"         → [0.891, -0.023, -0.456, ..., -0.678] # Different!
```

**How similarity is computed:**
```
Cosine Similarity = (A · B) / (||A|| × ||B||)

Where:
  A = query embedding
  B = document embedding
  Result: Score between -1 and 1
  Higher = more similar
```

### Vector Database - ChromaDB Internals

**Storage Structure:**
```
chroma_db/
├── chroma.sqlite3        # Metadata
└── index/                # HNSW index files
```

**HNSW (Hierarchical Navigable Small World) Index:**
- Graph-based algorithm for approximate nearest neighbor search
- Trade-off: Speed vs accuracy (default: 95% accuracy, 10x faster)
- Navigates through graph to find similar vectors

**Why not just compare all vectors?**
```
Brute force: O(n) - slow for large datasets
HNSW: O(log n) - fast even with millions of documents

For 319 documents: Both fast
For 1 million documents: HNSW is 1000x faster
```

### LLM - Ollama and llama3.2

**What happens during inference:**
1. Receives prompt (question + context)
2. Tokenizes text into tokens
3. Runs through transformer layers
4. Generates tokens one at a time
5. Stops when complete

**Why it's slow:**
- Running on CPU (no GPU)
- 3B parameter model
- Generates ~20 tokens/second on typical laptop
- Each token requires full forward pass

**Memory usage:**
- Model: ~2GB RAM
- Context: ~100MB per query
- Total: ~2.5GB RAM during query

## Components - Technical Stack

### 1. LangChain
**Purpose:** Orchestration framework
**What it does:**
- Provides abstractions for loaders, embeddings, vector stores
- Chains retrieval + generation steps
- Handles prompt construction

**Key classes used:**
```python
DirectoryLoader    # Load files from disk
TextLoader         # Parse text files
HuggingFaceEmbeddings  # Create embeddings
Chroma             # Vector database
Ollama             # LLM interface
RetrievalQA        # RAG chain
```

### 2. sentence-transformers
**Purpose:** Create embeddings
**Model:** all-MiniLM-L6-v2
**Details:**
- Based on Microsoft's MiniLM
- 22M parameters
- 384-dimensional output
- Optimized for semantic similarity
- Trained on 1B+ sentence pairs

### 3. ChromaDB
**Purpose:** Vector database
**Type:** Embedded database (runs in-process)
**Features:**
- HNSW index for fast search
- SQLite for metadata
- Persistent storage
- No separate server needed

### 4. Ollama
**Purpose:** Run LLMs locally
**Model:** llama3.2 (3B parameters)
**Architecture:**
- Runs as background service
- REST API for inference
- Model caching
- Automatic resource management

### 5. Kubernetes Examples
**Purpose:** Data source
**Content:**
- Production-ready YAML configs
- Documentation
- Best practices
- Maintained by K8s team

## What Kind of Questions to Ask

### Good Questions (Works Well)

**1. Configuration Examples**
```
"Show me a deployment with 3 replicas"
"How do I create a service?"
"Give me a StatefulSet example"
```
Why: Direct YAML examples exist in docs

**2. Concept Explanations**
```
"What's the difference between a Deployment and StatefulSet?"
"What is a headless service?"
"Explain persistent volumes"
```
Why: Documentation has conceptual content

**3. How-To Questions**
```
"How do I configure persistent storage?"
"How do I set resource limits?"
"How do I expose a service?"
```
Why: Examples show these patterns

**4. Troubleshooting**
```
"How do I check pod status?"
"How do I view logs?"
"How do I describe a resource?"
```
Why: Documentation includes kubectl commands

### Bad Questions (Won't Work Well)

**1. Questions Not in Documentation**
```
"What's the capital of France?"
"How do I write Python code?"
"What happened in the news today?"
```
Why: Not in K8s docs, system correctly says "I don't know"

**2. Very Specific Internal Info**
```
"What's our production cluster IP?"
"Show me our company's deployment"
```
Why: Needs your internal docs (add them to k8s-data/)

**3. Extremely New Features**
```
"How do I use the brand new feature released yesterday?"
```
Why: Documentation snapshot is static (re-clone to update)

**4. Opinion Questions**
```
"What's the best way to deploy?"
"Should I use X or Y?"
```
Why: LLM will give generic answer, not doc-specific

### Edge Cases to Test

**1. Ambiguous Questions**
```
"How do I scale?"
Expected: Asks for clarification or gives multiple options
```

**2. Partial Information**
```
"Configure storage"
Expected: Provides general guidance on PVs/PVCs
```

**3. Wrong Context**
```
"How do I deploy to AWS?"
Expected: Might not have AWS-specific info, gives generic K8s answer
```

## What Can Be Modified

### 1. Data Sources

**Current:**
```python
kubernetes/examples from GitHub
```

**Add More Sources:**
```python
# In demo.py, add more clones:

# Official K8s documentation
git clone https://github.com/kubernetes/website.git

# Prometheus docs
git clone https://github.com/prometheus/docs.git

# Your internal docs
# Just add YAML/Markdown to k8s-data/internal/
```

**Then update loader path:**
```python
loader = DirectoryLoader(
    './k8s-data/',  # Searches all subdirectories
    glob="**/*.yaml",
    ...
)
```

### 2. Chunk Size

**Current:** Uses default (varies by document)

**Modify:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
)

documents = text_splitter.split_documents(documents)
```

**When to adjust:**
- Larger chunks (2000+): Better context, slower search
- Smaller chunks (500): Faster search, less context
- More overlap (500): Better continuity, more storage

### 3. Number of Retrieved Documents

**Current:** k=3

**Modify:**
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Get top 5 instead of 3
)
```

**Trade-offs:**
- More docs (k=5-10): More context, slower, more noise
- Fewer docs (k=1-2): Faster, focused, might miss info

### 4. Embedding Model

**Current:** all-MiniLM-L6-v2 (384 dimensions)

**Other Options:**
```python
# Larger, more accurate (768 dimensions)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Smaller, faster (384 dimensions)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# Best quality (1024 dimensions, slow)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-roberta-large-v1"
)
```

**Note:** Changing model requires rebuilding index (delete chroma_db/)

### 5. LLM

**Current:** Ollama llama3.2 (3B)

**Upgrade to Larger Model:**
```bash
# Download larger model (better quality, slower)
ollama pull llama3.2:70b

# Use in code
llm = Ollama(model="llama3.2:70b")
```

**Use Paid API (Faster, Better):**
```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
```

### 6. Search Type

**Current:** Similarity search

**Other Options:**
```python
# Maximum Marginal Relevance (diverse results)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# Similarity with score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5}
)
```

### 7. Metadata Filtering

**Add metadata during loading:**
```python
from langchain.schema import Document

# Add metadata
for doc in documents:
    doc.metadata["source_type"] = "yaml" if doc.metadata["source"].endswith(".yaml") else "markdown"
    doc.metadata["category"] = "deployment" if "deployment" in doc.page_content.lower() else "other"
```

**Filter during retrieval:**
```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"source_type": "yaml"}  # Only YAML files
    }
)
```

### 8. Prompt Template

**Current:** Uses default RetrievalQA prompt

**Customize:**
```python
from langchain.prompts import PromptTemplate

template = """
You are a Kubernetes expert. Use the following documentation to answer the question.
If you don't know, say "I don't know" - don't make things up.

Documentation:
{context}

Question: {question}

Answer (provide YAML examples when relevant):
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
```

## Next Steps

### Immediate Improvements

**1. Add More Documentation**
```bash
cd k8s-data
git clone https://github.com/kubernetes/website.git
git clone https://github.com/prometheus/docs.git
```

**2. Add Your Internal Docs**
```bash
mkdir k8s-data/internal
# Copy your runbooks, postmortems, internal wikis
```

**3. Optimize Chunk Size**
```python
# Experiment with different sizes
chunk_sizes = [500, 1000, 2000]
# Test which gives best results
```

**4. Add Metadata Filtering**
```python
# Filter by file type, date, category
# Allows "search only YAML files" queries
```

### Production Considerations

**1. Better Vector Database**
```
ChromaDB (current) → Pinecone or Weaviate
Why: Better scaling, cloud-hosted, advanced features
```

**2. Add Caching**
```python
# Cache common queries
# Reduces latency and cost
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
```

**3. Evaluation Framework**
```python
# Test answer quality
# Compare different configurations
# Track metrics over time
```

**4. Add Hybrid Search**
```python
# Combine semantic + keyword search
# Better for exact matches (IDs, names)
```

**5. Build Web UI**
```python
# Streamlit or Gradio
# Better user experience
# Shareable with team
```

**6. Add Authentication**
```python
# If exposing as API
# Control access to internal docs
```

**7. Monitoring**
```python
# Log queries
# Track success rate
# Monitor performance
```

### Advanced Features

**1. Multi-Modal RAG**
```python
# Add diagrams, images
# Extract text from PDFs
# Process screenshots
```

**2. Agentic RAG**
```python
# LLM decides what to retrieve
# Multiple retrieval steps
# More complex reasoning
```

**3. Fine-Tuned Embeddings**
```python
# Train embeddings on your specific domain
# Better performance on internal jargon
```

**4. Query Rewriting**
```python
# LLM rephrases question for better retrieval
# Handles ambiguous queries
```

**5. Answer Citations**
```python
# Show which document each statement came from
# Helps verify accuracy
```

## Most Asked Questions

### Technical Questions

**Q: Why are embeddings 384 dimensions?**
A: That's the output size of the all-MiniLM-L6-v2 model. It's a balance between:
- Information captured (more dimensions = more nuance)
- Storage/speed (fewer dimensions = faster)
- 384 is sweet spot for most use cases

**Q: What's the difference between RAG and fine-tuning?**
A:
- Fine-tuning: Update model weights with your data (expensive, slow, permanent)
- RAG: Keep model as-is, just give it relevant docs (cheap, fast, flexible)
- RAG is better for: frequently changing info, factual accuracy, attribution
- Fine-tuning is better for: style, tone, domain-specific reasoning

**Q: Why use local LLM instead of GPT-4?**
A:
- Cost: $0 vs $0.01-0.10 per query
- Privacy: Data never leaves your machine
- Control: No rate limits, no API downtime
- Trade-off: Slower, less capable

**Q: How does semantic search differ from keyword search?**
A:
- Keyword: Finds exact words ("deployment" finds "deployment")
- Semantic: Finds meaning ("scaling" finds "deployment", "replicaset", "autoscaler")
- Semantic is better for: concepts, synonyms, related topics
- Keyword is better for: IDs, exact names, specific terms

**Q: What happens if my query matches nothing?**
A: System still retrieves top K documents (even with low scores), but LLM should say "I don't know" if context isn't relevant. This depends on LLM and prompt.

**Q: Can I use this for real-time data?**
A: No. RAG is for static documents. For real-time data, you'd need:
- Tool calling (LLM calls APIs)
- Streaming ingestion
- Real-time vector updates

**Q: How much RAM does this need?**
A:
- Embedding model: ~500MB
- LLM (llama3.2): ~2GB
- ChromaDB: ~100MB for 319 docs
- Total: ~3GB minimum

**Q: Can I run this on a server?**
A: Yes, but you'd need to:
- Expose as API (FastAPI)
- Handle concurrent requests
- Add authentication
- Scale vector database

### Usage Questions

**Q: What's a good chunk size?**
A: Depends on your documents:
- Short docs (YAML configs): 500-1000 chars
- Long docs (guides): 1000-2000 chars
- Very long docs: 2000+ chars
- Rule: Keep semantic units together (don't split mid-sentence)

**Q: How many documents can I add?**
A:
- Current setup: Thousands (fine)
- ChromaDB limit: Millions (with performance tuning)
- Practical limit: ~100K docs before you need better infrastructure

**Q: How often should I update the index?**
A:
- Static docs: Never
- Daily changes: Nightly rebuild
- Real-time: Need different architecture
- To update: Delete chroma_db/, re-run demo.py

**Q: Can I search multiple document types?**
A: Yes, loader handles YAML, Markdown, JSON, etc. Just add to k8s-data/

**Q: How do I know if my answer is correct?**
A: RAG doesn't guarantee correctness:
- Check sources provided
- Verify against official docs
- Test recommendations in safe environment
- Add evaluation framework for production

**Q: What if the retrieved chunks don't answer the question?**
A: LLM should say "I don't know." If it hallucinates:
- Adjust prompt to emphasize "only use provided context"
- Try different LLM (some are better at this)
- Increase K (retrieve more docs)

### Troubleshooting Questions

**Q: Why is my query slow?**
A: Check each step:
- Embedding: ~100ms (normal)
- Search: ~50ms (normal)
- LLM: 2-5 seconds (normal for local)
- If slower: Check CPU usage, model size

**Q: Why am I getting irrelevant results?**
A: Possible causes:
- Query too vague → Be more specific
- Wrong documents in index → Check data sources
- Embedding model mismatch → Use domain-appropriate model
- K too low → Increase retrieved documents

**Q: My index is too large. How do I reduce it?**
A:
- Remove unnecessary documents
- Increase chunk size (fewer chunks)
- Use smaller embedding model (fewer dimensions)
- Current 319 docs = ~50MB, very manageable

**Q: Can I use this offline?**
A: Yes, after initial setup:
- Ollama runs locally
- ChromaDB is local
- No internet needed for queries
- Only need internet to download docs/models initially

### Business Questions

**Q: What does this cost in production?**
A:
- Free tier (current setup): $0
- Light usage with OpenAI: ~$10-50/month
- Production with dedicated infra: $500-5000/month
- Depends on: query volume, model choice, infrastructure

**Q: How accurate is this?**
A: Depends on:
- Quality of source documents (good docs = good answers)
- Relevance of retrieved chunks (better retrieval = better answers)
- LLM capabilities (GPT-4 > llama3.2)
- Typical: 70-90% accuracy on answerable questions

**Q: Can this replace our documentation?**
A: No, it's a search tool, not a replacement:
- Use for: Quick answers, discovery, troubleshooting
- Don't use for: Source of truth, compliance, exact specifications
- Think: Enhanced search, not documentation replacement

**Q: How do I measure success?**
A: Track metrics:
- User satisfaction (thumbs up/down)
- Query success rate (got useful answer?)
- Time saved vs manual search
- Coverage (% of queries answerable)

**Q: What about hallucinations?**
A: RAG reduces hallucinations but doesn't eliminate them:
- Always provide sources
- Encourage users to verify
- Use better LLMs (they hallucinate less)
- Adjust prompt to emphasize "only use context"

## Summary

**What RAG Does:**
Finds relevant documents, sends them to LLM, gets answer

**How It Works:**
Embeddings → Vector search → LLM generation

**What Makes It Good:**
- Semantic understanding (meaning, not just keywords)
- Always up-to-date (add new docs anytime)
- Cites sources (verifiable)
- Cheap (free tier works well)

**What Makes It Limited:**
- Only as good as source documents
- Can't reason beyond what's in docs
- Local LLM is slow
- No real-time data

**Best For:**
- Searching documentation
- Finding examples
- Answering factual questions
- Internal knowledge bases

**Not For:**
- Real-time data
- Complex reasoning without docs
- Generating novel solutions
- Production-critical decisions without verification
