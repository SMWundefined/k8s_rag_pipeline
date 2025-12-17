# GenAI 101: RAG for Kubernetes Docs
## Comprehensive Slide Deck Guide

---

# SECTION 1: THE PROBLEM & SOLUTION

## Slide 1: The Problem

**Title:** "Finding the Right K8s Config is Hard"

**Talking Points:**
- Kubernetes has 100+ resource types
- Documentation is scattered across multiple sites
- YAML syntax is easy to get wrong
- "I know it exists, but I can't find it"

**Visual:** Screenshot of overwhelming K8s documentation

---

## Slide 2: Traditional Search Fails

**Title:** "Keyword Search Doesn't Understand You"

| You Search For | What You Want | What grep Finds |
|----------------|---------------|-----------------|
| "scale my app" | Deployment replicas | Nothing (no exact match) |
| "persistent storage" | PVC examples | Random mentions of "persistent" |
| "expose service" | Service + Ingress | Files with "expose" in comments |

**Talking Point:** Keyword search finds words, not meaning.

---

## Slide 3: The Solution - RAG

**Title:** "RAG: Search by Meaning, Not Keywords"

**What is RAG?**
- **R**etrieval - Find relevant documents
- **A**ugmented - Add them to the prompt
- **G**eneration - LLM creates the answer

**Result:** Ask "How do I scale my app?" → Get actual deployment YAML with replicas config

---

# SECTION 2: HOW IT WORKS

## Slide 4: Architecture Overview

**Title:** "The RAG Pipeline"

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  K8s Docs   │───▶│ Embeddings  │───▶│  ChromaDB   │
│  (YAML/MD)  │    │ (HuggingFace)│    │(Vector Store)│
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
│   Answer    │◀───│    LLM      │◀───│  Retriever  │◀── Question
│ + Sources   │    │  (Ollama)   │    │  (Top-K)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Talking Points:**
1. Documents are converted to vectors (numbers that capture meaning)
2. Question is also converted to a vector
3. Find documents with similar vectors
4. Send relevant docs + question to LLM
5. LLM generates answer based on actual documentation

---

## Slide 5: What Are Embeddings?

**Title:** "Embeddings: Meaning as Numbers"

**Concept:**
- Text → 384 numbers (vector)
- Similar meaning = similar vectors

**Example:**
```
"deployment"  → [0.023, -0.145, 0.089, ...]
"replicaset"  → [0.028, -0.141, 0.092, ...]  ← Similar!
"coffee"      → [0.891, -0.023, -0.456, ...]  ← Different!
```

**Visual:** 2D plot showing similar concepts clustered together

**Talking Point:** This is why searching "scale my app" finds "deployment replicas" - they have similar vectors even without matching words.

---

## Slide 6: Vector Database

**Title:** "ChromaDB: Finding Similar Vectors Fast"

**What it does:**
- Stores document vectors
- Finds nearest neighbors in milliseconds
- Uses HNSW algorithm (graph-based search)

**Why not compare all vectors?**
| Method | 1,000 docs | 1,000,000 docs |
|--------|------------|----------------|
| Brute force | 10ms | 10 seconds |
| HNSW | 1ms | 10ms |

**Talking Point:** Even with millions of documents, search is near-instant.

---

## Slide 7: The LLM's Role

**Title:** "LLM: From Context to Answer"

**Input (Prompt):**
```
Context:
[deployment.yaml content]
[scaling-guide.md content]

Question: How do I scale my app to 3 replicas?

Answer:
```

**Output:**
```
To scale your app to 3 replicas, modify your deployment:

apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3  # ← Change this value
```

**Talking Point:** The LLM doesn't guess - it synthesizes an answer from the actual docs we retrieved.

---

# SECTION 3: HISTORY & CONTEXT

## Slide 8: The Evolution of Search

**Title:** "From Keywords to Understanding"

| Era | Technology | Example |
|-----|------------|---------|
| 1990s | Keyword Search | grep, ctrl+F |
| 2000s | Indexed Search | Google, Elasticsearch |
| 2010s | Semantic Search | Word2Vec, BERT |
| 2020s | RAG | Embeddings + LLMs |

**Key Milestone:** 2020 - Facebook AI publishes "Retrieval-Augmented Generation" paper

**Talking Point:** RAG combines 30 years of search evolution with modern LLMs.

---

## Slide 9: Why RAG Over Fine-Tuning?

**Title:** "RAG vs Fine-Tuning: Choose Wisely"

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Cost | Free (local) | $100s-$1000s |
| Update data | Add files, done | Retrain model |
| Accuracy | Cites sources | May hallucinate |
| Speed to deploy | Minutes | Hours/days |
| Best for | Facts, docs | Style, tone |

**Talking Point:** RAG is perfect for documentation search - it's fast, cheap, and verifiable.

---

## Slide 10: The Tech Stack (100% Free)

**Title:** "Everything Runs Locally"

| Component | Tool | Cost | What It Does |
|-----------|------|------|--------------|
| LLM | Ollama (llama3.2) | $0 | Generates answers |
| Embeddings | HuggingFace | $0 | Text → vectors |
| Vector DB | ChromaDB | $0 | Stores & searches |
| Framework | LangChain | $0 | Glues it together |

**Total Cost: $0**

**Talking Point:** No API keys, no cloud bills, no data leaving your machine.

---

# SECTION 4: THE CODE WALKTHROUGH

## Slide 11: The Complete Code (~60 lines)

**Title:** "Simple Enough to Fit on a Slide"

```python
# 1. Load documents
loader = DirectoryLoader('./k8s-data/', glob="**/*.yaml")
documents = loader.load()

# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Store in vector database
vectorstore = Chroma.from_documents(documents, embeddings)

# 4. Connect to LLM
llm = OllamaLLM(model="llama3.2")

# 5. Create RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# 6. Ask questions!
answer = rag_chain.invoke("Show me a deployment example")
```

---

## Slide 12: Live Demo

**Title:** "Let's Try It"

**Demo Questions:**
1. "Show me a deployment with 3 replicas"
2. "How do I configure persistent storage?"
3. "What's a headless service?"

**Expected Output:**
```
Question: Show me a deployment example

Searching...

Answer: Here's a basic Kubernetes deployment...
[YAML example from actual docs]

Sources:
  - ./k8s-data/examples/web/nginx-deployment.yaml
  - ./k8s-data/examples/databases/redis-deployment.yaml
```

**Talking Point:** Notice it shows the source files - you can verify the answer!

---

# SECTION 5: WHAT YOU CAN CUSTOMIZE

## Slide 13: Customization Options

**Title:** "Make It Your Own"

| What | How | When to Change |
|------|-----|----------------|
| Data sources | Add files to `k8s-data/` | Add internal docs, runbooks |
| Chunk size | `chunk_size=1000` | Long vs short documents |
| Results count | `k=3` to `k=10` | Need more context |
| Embedding model | Change model name | Better accuracy needed |
| LLM | Swap Ollama model | Faster/smarter responses |
| Prompt | Edit template | Custom answer format |

---

## Slide 14: Add Your Own Docs

**Title:** "Extend with Internal Knowledge"

```bash
# Add your internal docs
mkdir k8s-data/internal
cp ~/runbooks/*.md k8s-data/internal/
cp ~/postmortems/*.md k8s-data/internal/

# Re-run the demo
rm -rf chroma_db/  # Clear old index
python demo_simple.py
```

**Now you can ask:**
- "What was the cause of last week's outage?"
- "How do we deploy to production?"
- "What's our rollback procedure?"

---

## Slide 15: Upgrade the LLM

**Title:** "Better Models, Better Answers"

**Local (Free):**
```bash
ollama pull llama3.2:70b  # Larger, smarter
ollama pull codellama     # Better for code
ollama pull mistral       # Fast and capable
```

**Cloud (Paid, Faster):**
```python
# OpenAI (~$0.01 per query)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet")
```

---

## Slide 16: Change the Prompt

**Title:** "Control the Output Format"

```python
prompt = PromptTemplate.from_template("""
You are a Kubernetes expert at [Company Name].

Rules:
- Only use information from the provided context
- Always include YAML examples when relevant
- If you don't know, say "I don't know"
- Keep answers concise

Context: {context}

Question: {question}

Answer:
""")
```

**Talking Point:** The prompt is where you inject your company's style and requirements.

---

# SECTION 6: NEXT STEPS & FUTURE

## Slide 17: Immediate Next Steps

**Title:** "What to Do After This Workshop"

1. **Add more documentation**
   ```bash
   git clone https://github.com/kubernetes/website.git k8s-data/website
   ```

2. **Add internal docs**
   - Runbooks, postmortems, internal wikis

3. **Experiment with settings**
   - Try k=5 instead of k=3
   - Test different chunk sizes

4. **Build a simple UI**
   - Streamlit or Gradio (10 lines of code)

---

## Slide 18: Production Considerations

**Title:** "From Demo to Production"

| Demo | Production |
|------|------------|
| ChromaDB (local) | Pinecone, Weaviate (cloud) |
| Single user | Auth + rate limiting |
| No caching | Redis cache for common queries |
| No monitoring | Logging, metrics, alerts |
| CLI interface | Web UI or API |

**Talking Point:** This demo is ~80% of production code. The remaining 20% is infrastructure.

---

## Slide 19: Advanced Features (Future)

**Title:** "Where RAG is Heading"

**Hybrid Search**
- Combine semantic + keyword search
- Better for exact matches (IDs, names)

**Multi-Modal RAG**
- Include images, diagrams, PDFs
- "Show me the architecture diagram for service X"

**Agentic RAG**
- LLM decides what to retrieve
- Multiple retrieval steps
- More complex reasoning

**Query Rewriting**
- LLM rephrases vague questions
- "How do I do the thing?" → "How do I scale a deployment?"

---

## Slide 20: Evaluation & Metrics

**Title:** "How Do You Know It's Working?"

**Metrics to Track:**
| Metric | How to Measure |
|--------|----------------|
| Retrieval accuracy | Are the right docs being found? |
| Answer quality | User ratings (thumbs up/down) |
| Latency | Time from question to answer |
| Coverage | % of questions answered |

**Simple Evaluation:**
```python
test_questions = [
    ("How do I create a deployment?", "deployment.yaml"),
    ("What is a PVC?", "pvc"),
]
# Check if expected doc is in top-3 results
```

---

# SECTION 7: Q&A PREP

## Slide 21: Anticipated Questions

**Title:** "FAQ"

**Q: Why is it slow?**
A: Local LLM on CPU. Use GPU or cloud API for speed.

**Q: Can it hallucinate?**
A: Less than pure LLM, but yes. Always check sources.

**Q: How much RAM does it need?**
A: ~3GB (500MB embeddings + 2GB LLM + 500MB index)

**Q: Can I use this for sensitive docs?**
A: Yes! Everything runs locally. No data leaves your machine.

**Q: How often should I update the index?**
A: Whenever docs change. Delete `chroma_db/` and re-run.

---

## Slide 22: Summary

**Title:** "Key Takeaways"

1. **RAG = Retrieval + Augmented + Generation**
   - Find relevant docs → Send to LLM → Get answer

2. **Embeddings capture meaning**
   - Similar concepts have similar vectors

3. **100% free and local**
   - No API keys, no cloud, no data leakage

4. **Easy to extend**
   - Add your docs, change the model, customize the prompt

5. **Production-ready pattern**
   - Same architecture powers enterprise search tools

---

## Slide 23: Resources

**Title:** "Learn More"

**Documentation:**
- [LangChain Docs](https://python.langchain.com/)
- [Ollama](https://ollama.ai)
- [ChromaDB](https://docs.trychroma.com/)

**Papers:**
- [Original RAG Paper (2020)](https://arxiv.org/abs/2005.11401)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)

**This Workshop:**
- GitHub: `[your-repo-url]`
- Demo: `python demo_simple.py`

---

## Slide 24: Thank You

**Title:** "Questions?"

```
┌─────────────────────────────────────────┐
│                                         │
│   "The best way to predict the future   │
│    is to implement it."                 │
│                                         │
│                    - David Heinemeier   │
│                                         │
└─────────────────────────────────────────┘
```

**Contact:** [Your email/slack]

---

# APPENDIX: SPEAKER NOTES

## Timing Guide (40 minutes)

| Section | Slides | Time |
|---------|--------|------|
| Problem & Solution | 1-3 | 5 min |
| How It Works | 4-7 | 8 min |
| History & Context | 8-10 | 5 min |
| Code Walkthrough | 11-12 | 8 min |
| Customization | 13-16 | 6 min |
| Next Steps | 17-20 | 5 min |
| Q&A | 21-24 | 3 min |

## Demo Checklist

Before the workshop:
- [ ] Ollama running (`ollama serve`)
- [ ] Model downloaded (`ollama pull llama3.2`)
- [ ] Virtual env activated (`source venv/bin/activate`)
- [ ] Test query works (`python demo_simple.py`)
- [ ] Terminal font size increased (for visibility)

## Backup Plans

**If Ollama fails:**
- Show pre-recorded demo video
- Walk through code without live execution

**If questions go off-track:**
- "Great question! Let's discuss after the session"
- Redirect to the FAQ slide

**If time runs short:**
- Skip slides 8-10 (History)
- Skip slides 17-20 (Future)
- Go straight to demo + Q&A

---

# APPENDIX: GLOSSARY

| Term | Definition |
|------|------------|
| **Embedding** | Vector representation of text that captures semantic meaning |
| **Vector** | A list of numbers (e.g., [0.1, -0.3, 0.5, ...]) |
| **HNSW** | Hierarchical Navigable Small World - fast approximate nearest neighbor algorithm |
| **RAG** | Retrieval-Augmented Generation - combining search with LLM |
| **LLM** | Large Language Model (GPT, Claude, Llama, etc.) |
| **Chunk** | A piece of a document (e.g., 1000 characters) |
| **Retriever** | Component that finds relevant documents |
| **Prompt** | The text sent to the LLM (includes context + question) |
| **LCEL** | LangChain Expression Language - modern way to build chains |
| **Inference** | Running the model to get a prediction/answer |

---

# APPENDIX: TROUBLESHOOTING

## Common Issues

**"Ollama not found"**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # In one terminal
ollama pull llama3.2  # In another terminal
```

**"No module named langchain"**
```bash
source venv/bin/activate  # Must activate venv first!
pip install -r requirements.txt
```

**"No files found"**
```bash
ls k8s-data/examples/  # Check if data exists
python demo.py  # Run full demo first to download data
```

**Slow responses**
- Normal for local LLM (2-5 seconds)
- Use GPU if available
- Or switch to OpenAI API

**Out of memory**
- Close other applications
- Use smaller model: `ollama pull llama3.2:1b`
- Reduce chunk count: `k=2` instead of `k=3`
