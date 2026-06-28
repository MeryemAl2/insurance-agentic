# Video Script - 2 Minutes Maximum

## 0:00 - 0:20
Introduce the project:

> This project is an Agentic RAG assistant for insurance questions. It uses a local document base, ChromaDB vector search, and a manually built LangGraph workflow instead of LangChain `create_agent`.

Show the folders: `data/`, `vectorstore/`, `tools/`, `graph/`, `evaluation/`.

## 0:20 - 0:45
Show the graph:

> The graph starts by analyzing the user question, retrieves relevant documents, grades their relevance, rewrites the query if evidence is weak, summarizes the retrieved passages, generates an answer, evaluates it, then saves memory.

Open `docs/graph.svg` or `docs/graph.mmd`.

## 0:45 - 1:25
Run the app:

```bash
streamlit run app.py
```

Ask one simple question:

> What documents are usually needed to file a motor insurance claim?

Then open the Trace section and show the retrieved sources, relevance score, evaluation score, and response time.

## 1:25 - 1:45
Ask one complex question:

> If my car was damaged while being used for paid delivery work, how should I determine whether my motor policy covers the claim?

Explain that the answer is grounded in retrieved document excerpts and warns when policy wording is required.

## 1:45 - 2:00
Show evaluation:

```bash
python evaluation/evaluate.py
```

> The evaluation uses 10 simple and 10 complex questions and records answer quality, response time, and retrieved document relevance in `evaluation/results/latest_results.json`.
