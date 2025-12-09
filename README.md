# GenAI 101: Build an AI Search for K8s Docs

**40-minute hands-on workshop** teaching you to build a RAG (Retrieval Augmented Generation) system for searching Kubernetes documentation using natural language.
## What You'll Build

Ask questions like:
- "Show me a deployment with 3 replicas"
- "How do I configure persistent storage?"
- "What's the difference between a Service and an Ingress?"

Get back actual K8s YAML configs + explanations!

## Quick Start

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai) (free, local LLM)

### Setup (5 minutes)

```bash
# 1. Clone this repo
git clone https://github.com/[YOUR-USERNAME]/k8s-rag-workshop.git
cd k8s-rag-workshop

# 2. Install Ollama (if not already installed)
# Visit https://ollama.ai or run:
curl -fsSL https://ollama.com/install.sh | sh

# 3. Run setup script
chmod +x setup.sh
./setup.sh

# 4. Run the demo!
source venv/bin/activate
python demo.py
```

That's it! Start asking questions about Kubernetes.

## What's Inside

```
k8s-rag-workshop/
├── demo.py              # Main demo (simple, ~60 lines)
├── setup.sh             # One-command setup
├── requirements.txt     # Python dependencies
├── sample-data/         # Sample K8s configs (included!)
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── statefulset.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   └── k8s-basics.md
└── README.md            # This file
```

## How It Works
### The Process

1. **Load** K8s configs and docs
2. **Convert** to embeddings (meaning as numbers)
3. **Store** in vector database (ChromaDB)
4. **Query**: Find similar docs → Send to LLM → Get answer

### Why This is Cool

**Traditional search (grep):**
- Finds exact keywords only
- Doesn't understand relationships
- Can't answer questions

**RAG:**
- Understands meaning
- Knows "deployment" relates to "replicaset"
- Returns relevant configs even without exact matches
- Answers in natural language

## Tech Stack

### What We Use (100% Free)

| Component | Tool | Cost |
|-----------|------|------|
| LLM | Ollama (llama3.2) | $0 |
| Embeddings | HuggingFace (sentence-transformers) | $0 |
| Vector DB | ChromaDB (local) | $0 |
| Framework | LangChain | $0 |

**Total cost: $0** 

## Usage

### Basic Usage

```bash
python demo.py
```

Then ask questions:
```
❓ Your question: Show me a deployment
✅ Answer: Here's a deployment example from deployment.yaml...

❓ Your question: How do I configure persistent storage?
✅ Answer: You can use PersistentVolumeClaims...
```

### Adding Your Own Docs

1. Add YAML or Markdown files to `sample-data/`
2. Run `python demo.py` again
3. That's it! Your new docs are now searchable

## Workshop Structure

If you're attending the workshop:

**Slides (8 min)**
- Problem & Solution
- How RAG works
- Live demo

**Live Coding (28 min)**
- Walk through the code
- Explain each concept
- See it work

**Interactive (4 min)**
- Try your own queries
- Experiment
- Discuss results

## Troubleshooting

### "Ollama not found"
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2
```

### "No files found in ./sample-data/"
Make sure you're in the workshop directory:
```bash
ls sample-data/  # Should see YAML files
```

### Python errors
```bash
# Recreate environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Slow responses
This is normal for local LLMs! Expect 2-5 seconds per query.
Want faster? Upgrade to OpenAI (see Optional Upgrades above).

## Learn More

### Concepts

**Embeddings**: Numbers that capture meaning. Similar meanings = similar numbers.

**Vector Database**: Stores embeddings, finds similar ones fast.

**RAG**: Retrieval (find docs) + Augmented (add context) + Generation (LLM answer).

### Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama](https://ollama.ai)
- [ChromaDB](https://docs.trychroma.com/)
- [Original RAG Paper](https://arxiv.org/abs/2005.11401)


### Production Considerations

- Use better vector stores (Pinecone, Weaviate) for scale
- Add caching for common queries
- Implement evaluation metrics
- Add authentication if exposing as API

## Contributing

Found a bug? Have an improvement? PRs welcome!

## License

MIT License - use freely for learning and teaching!

