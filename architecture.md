# K8s RAG Pipeline Architecture

```mermaid
flowchart LR
    subgraph Input["Data Sources"]
        K8S["K8s YAML Configs<br/>& Markdown Docs"]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        LOADER["Document Loader<br/>(LangChain)"]
        EMBED["Embeddings<br/>(HuggingFace)"]
    end

    subgraph Storage["Vector Store"]
        CHROMA[("ChromaDB")]
    end

    subgraph Query["Query Pipeline"]
        USER["User Question"]
        RETRIEVER["Retriever<br/>(Top-K Similar)"]
        LLM["Local LLM<br/>(Ollama/Llama3.2)"]
        ANSWER["Generated Answer<br/>+ YAML Examples"]
    end

    K8S --> LOADER
    LOADER --> EMBED
    EMBED --> CHROMA

    USER --> RETRIEVER
    CHROMA --> RETRIEVER
    RETRIEVER --> LLM
    LLM --> ANSWER

    style Input fill:transparent,stroke:#01579b
    style Ingestion fill:transparent,stroke:#e65100
    style Storage fill:transparent,stroke:#7b1fa2
    style Query fill:transparent,stroke:#2e7d32
```

## Simplified Version (for slides)

```mermaid
flowchart LR
    A["K8s Docs"] --> B["Embeddings<br/>(HuggingFace)"]
    B --> C[("ChromaDB")]

    D["Question"] --> E["Retriever"]
    C --> E
    E --> F["LLM<br/>(Ollama)"]
    F --> G["Answer"]

    style A fill:transparent,stroke:#1976d2
    style B fill:transparent,stroke:#f57f17
    style C fill:transparent,stroke:#8e24aa
    style D fill:transparent,stroke:#388e3c
    style E fill:transparent,stroke:#388e3c
    style F fill:transparent,stroke:#c62828
    style G fill:transparent,stroke:#388e3c
```

## Tech Stack Summary

| Component | Tool |
|-----------|------|
| LLM | Ollama (llama3.2) |
| Embeddings | HuggingFace (sentence-transformers) |
| Vector DB | ChromaDB |
| Framework | LangChain |
