# RAG System with Retrieval Benchmarking
## GCPL AI Hackathon — Option B

## Overview
A Retrieval-Augmented Generation (RAG) system built on FMCG 
and supply chain data, with comparative evaluation of two 
chunking strategies.

## Architecture
```
Wikipedia Articles → Text Chunks → Embeddings → ChromaDB
                                                    ↓
User Question → Embedding → Similarity Search → Top 3 Chunks
                                                    ↓
                                    Groq LLM → Final Answer
```

## Chunking Strategies Compared
| Strategy | Chunk Size | Overlap | Score |
|---|---|---|---|
| Strategy 1 | 500 chars | 50 chars | 83% |
| Strategy 2 | 200 chars | 100 chars | 73% |

## Tech Stack
- **LLM:** Groq (llama-3.1-8b-instant)
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 (local)
- **Vector DB:** ChromaDB
- **Framework:** LangChain
- **Data:** Wikipedia (8 FMCG/AI articles)

## How to Run
1. Clone the repo
2. Install dependencies:
pip install langchain langchain-groq langchain-huggingface 
chromadb python-dotenv wikipedia-api
3. Add GROQ_API_KEY to .env file
4. Fetch data: python fetch_data.py
5. Run RAG system: python app.py
6. Run evaluation: python evaluate.py

## Results
Strategy 1 (large chunks) outperformed Strategy 2 (small chunks)
with 83% vs 73% accuracy across 10 test queries.
Larger chunks preserve semantic context better for 
factual Q&A on FMCG topics.

## Known Limitations
- Dataset limited to Wikipedia summaries
- Scoring is keyword-based, not semantic
- No reranking implemented
- Small dataset size limits generalization