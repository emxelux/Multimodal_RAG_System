"""
End-to-end RAG evaluation for ChatPDF.

For every unique source PDF referenced in evaluation/questions.json:
    1. Ingest + chunk + index it ONCE under a fresh document_id.
For every question:
    2. retrieve_context -> rerank_results -> build doc_context, exactly
       the way app/main.py's /generation endpoint does it.
    3. Generate an answer with the same LLM the app uses.

Results are scored with ragas (faithfulness, answer relevancy,
context precision, context recall) using your own Groq model +
HF embeddings as the judge, so no OpenAI key is required.

Run with:  python -m evaluation.rag_evaluation
(from the project root, so the "app"/"data_preprocessing"/"llm" imports resolve)
"""

import json
import os
import uuid

from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm

load_dotenv()

from data_preprocessing.ingest import ingest_pdf, build_documents
from data_preprocessing.chunking import split_markdown_document
from data_preprocessing.vector_db import (
    upsert_split_documents,
    retrieve_context,
    rerank_results,
)
from llm.ask_llm import generation

EVAL_USER_ID = "8a707bc8-2b26-4033-a775-e2b0c2bd47c4"
DOCUMENTS_DIR = "evaluation/Documents"
QUESTIONS_PATH = "evaluation/questions.json"
RAW_RESULTS_PATH = "evaluation/eval_raw_results.json"
SCORES_PATH = "evaluation/eval_scores.csv"
TOP_K_RETRIEVE = 10
TOP_N_RERANK = 3


def _field(r, key, default=None):
   
    if isinstance(r, dict):
        return r.get(key, default)

    metadata = getattr(r, "metadata", None)
    if key == "content":
        page_content = getattr(r, "page_content", None)
        if page_content is not None:
            return page_content
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]

    if hasattr(r, key):
        return getattr(r, key)

    if hasattr(r, "model_dump"):
        try:
            return r.model_dump().get(key, default)
        except Exception:
            pass

    return default


def build_doc_context(final_context):
    """Same formatting main.py feeds to the LLM. Note: chunks store their
    source under metadata['source'] (see vector_db.upsert_split_documents),
    not 'source_name' -- falling back to 'source' avoids empty citations."""
    return "\n\n".join(
        f"[Chunk {i+1}] Source: {_field(r, 'source', _field(r, 'source_name', ''))}, "
        f"Page: {_field(r, 'page', '?')}\n{_field(r, 'content', '')}"
        for i, r in enumerate(final_context)
    )


def index_documents(filenames):
    """Ingest + chunk + upsert each unique PDF exactly once.
    Returns {filename: document_id} so questions can look up their doc."""
    doc_ids = {}
    for filename in tqdm(filenames, desc="Indexing documents"):
        ingested = ingest_pdf(f"{DOCUMENTS_DIR}/{filename}")
        docs = build_documents(ingested, filename)
        nodes = split_markdown_document(docs)

        document_id = str(uuid.uuid4())
        upsert_split_documents(nodes, user_id=EVAL_USER_ID, source_document=document_id)

        doc_ids[filename] = document_id
        logger.info(f"Indexed '{filename}' -> {len(nodes)} chunks, document_id={document_id}")

    return doc_ids


def run_pipeline():
    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    unique_sources = sorted({q["source"] for q in questions})
    doc_ids = index_documents(unique_sources)

    rows = []
    for q in tqdm(questions, desc="Retrieval + generation"):
        document_id = doc_ids[q["source"]]

        retrieved = retrieve_context(
            query=q["user_input"],
            user_id=EVAL_USER_ID,
            source_document=document_id,
            top_k=TOP_K_RETRIEVE,
        )
        final_context = rerank_results(q["user_input"], retrieved, top_n=TOP_N_RERANK)
        doc_context = build_doc_context(final_context)

        answer = generation(query=q["user_input"], doc_context=doc_context)

        rows.append(
            {
                "user_input": q["user_input"],
                "response": answer,
                "retrieved_contexts": [_field(r, "content", "") for r in final_context],
                "reference": q["reference"],
                "source": q["source"],
            }
        )

    return rows


def get_ragas_judge():
    from langchain.chat_models import init_chat_model
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    eval_llm = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
    eval_embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-large-en-v1.5",
        huggingfacehub_api_token=os.getenv["HF_TOKEN"],
    )
    return LangchainLLMWrapper(eval_llm), LangchainEmbeddingsWrapper(eval_embeddings)


def score_with_ragas(rows):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    judge_llm, judge_embeddings = get_ragas_judge()

    dataset = Dataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
            for r in rows
        ]
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    return result


if __name__ == "__main__":
    rows = run_pipeline()

    with open(RAW_RESULTS_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Saved raw retrieval/generation results -> {RAW_RESULTS_PATH}")

    scores = score_with_ragas(rows)
    df = scores.to_pandas()
    df.to_csv(SCORES_PATH, index=False)
    logger.info(f"Saved per-question scores -> {SCORES_PATH}")

    print("\n=== Aggregate RAGAS scores ===")
    print(scores)

    print("\n=== Per-source breakdown (mean faithfulness / answer_relevancy) ===")
    df["source"] = [r["source"] for r in rows]
    (
        df.groupby("source")[["faithfulness", "answer_relevancy"]].mean().round(3)
    )