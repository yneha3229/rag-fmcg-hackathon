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
1. Clone the repo and install the dependencies:
   ```
   git clone https://github.com/yneha3229/rag-fmcg-hackathon.git
   cd rag-fmcg-hackathon
   pip install -r requirements.txt
   ```
2. Create a `.env` file containing your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Fetch the data, run the RAG system, and run the evaluation:
   ```
   python fetch_data.py
   python app.py
   python evaluate.py
   ```

## Results
Strategy 1 (large chunks) outperformed Strategy 2 (small chunks)
with a keyword-match score of 83% vs 73% across 10 test queries.
Larger chunks preserve semantic context better for 
factual Q&A on FMCG topics.

## Known Limitations
- Dataset limited to Wikipedia summaries
- Scoring is keyword-based, not semantic
- No reranking implemented
- Small dataset size limits generalization