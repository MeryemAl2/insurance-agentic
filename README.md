# Insurance Agentic RAG

Manual LangGraph RAG assistant for insurance documents covering motor, health, home, claims, FAQ, and regulations.

## Project Shape

```text
insurance-agentic-rag/
|-- app.py
|-- graph/
|   |-- state.py
|   |-- nodes.py
|   |-- graph.py
|   `-- prompts.py
|-- tools/
|   |-- retriever.py
|   |-- grader.py
|   |-- rewrite.py
|   |-- summarizer.py
|   `-- evaluator.py
|-- vectorstore/
|   |-- ingest.py
|   `-- chroma_db/
|-- data/
|-- docs/
|-- evaluation/
`-- requirements.txt
```

## Graph Flow

```text
START
-> analyze_question
-> retrieve_documents
-> grade_documents
-> if weak: rewrite_question -> retrieve_documents
-> summarize_documents
-> generate_answer
-> self_evaluate_answer
-> save_memory
-> END
```

## 1. Create Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## 2. Configure LLM

For a free smoke test with no API key:

```env
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=hash
```

This tests the full app flow, but the answer quality is intentionally basic.

For OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
```

For Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-3.1-flash-lite
```

For Grok / xAI:

```env
LLM_PROVIDER=xai
XAI_API_KEY=your_xai_key
XAI_MODEL=grok-4.3
```

For a local Ollama model:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

The default `EMBEDDING_PROVIDER=hash` works for smoke tests with no external embedding service. For better retrieval, switch to `openai` or `sentence_transformers`.

## 3. Add Or Replace Documents

Sample `.txt` files are included under `data/` so the project can be ingested immediately. Replace them with real `.txt`, `.md`, or `.pdf` documents when ready.

## 4. Ingest Documents

```bash
python vectorstore/ingest.py
```

## 5. Run App

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit and ask an insurance question.

## 6. Run Evaluation

```bash
python evaluation/evaluate.py
```

The evaluator runs 10 simple and 10 complex questions. It records answer quality, response time, and retrieved document relevance in `evaluation/results/latest_results.json`.

## Deliverables For The Evaluation

- Source code: this repository.
- Graph visualization: `docs/graph.svg` and `docs/graph.mmd`.
- Individual report draft: `docs/report.md`.
- PDF report: `docs/rapport_individuel.pdf`.
- PDF export command: `python scripts/export_report_pdf.py`.
- Video guide: `docs/video_script.md`.
- Evaluation questions: `evaluation/questions_simple.json` and `evaluation/questions_complex.json`.
- Evaluation results: `evaluation/results/latest_results.json`.

## GitHub Publication

```bash
git init
git add .
git commit -m "Initial insurance agentic rag project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/insurance-agentic-rag.git
git push -u origin main
```

Do not commit `.env`; use `.env.example` for configuration examples.

## Notes

- The graph is built manually with LangGraph `StateGraph`.
- LangChain agents are not used.
- ChromaDB stores document chunks under `vectorstore/chroma_db/`.
- Streamlit session state stores conversation memory, and LangGraph uses an in-memory checkpointer.
