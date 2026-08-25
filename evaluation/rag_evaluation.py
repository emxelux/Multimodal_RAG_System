import json
import os

from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from data_preprocessing.ingest import (
    ingest_pdf,
    build_documents,
)

from data_preprocessing.chunking import (
    split_markdown_document,
)

from data_preprocessing.vector_db import (
    retrieve_context,
    rerank_results,
    upsert_split_documents,
)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY is not set in the environment or .env file."
    )


# ============================================================
# 2. INITIALIZE LLM
# ============================================================

llm = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

USER_ID = "854e73ae-f2d8-42e6-9c6d-a2a88e27a9d5"

DOCUMENTS_DIR = "evaluation/Documents"

GROUND_TRUTH_FILE = (
    "evaluation/Documents/ground_truth.json"
)

OUTPUT_FILE = (
    "evaluation/Documents/ground_truth_with_answers.json"
)

SYSTEM_PROMPT_FILE = (
    "prompts/system_prompt_v2.txt"
)


# ============================================================
# 4. DOCUMENTS TO EVALUATE
# ============================================================

documents = [
    "DATA-ANALYSIS-REPORT-TEAM-5.pdf",
    "Medical Paper on Cancer.pdf",
    "Technical Paper on Machine Learning.pdf",
]


# ============================================================
# 5. LOAD GROUND TRUTH
# ============================================================

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8",
) as f:

    ground_truth = json.load(f)


# ============================================================
# 6. LOAD SYSTEM PROMPT
# ============================================================

with open(
    SYSTEM_PROMPT_FILE,
    "r",
    encoding="utf-8",
) as f:

    system_prompt = f.read()


# ============================================================
# 7. INDEX DOCUMENTS ONCE
# ============================================================

print("\n" + "=" * 60)
print("INDEXING EVALUATION DOCUMENTS")
print("=" * 60)


for document in tqdm(
    documents,
    desc="Indexing documents",
):

    document_path = os.path.join(
        DOCUMENTS_DIR,
        document,
    )

    print(
        f"\nProcessing: {document}"
    )


    # --------------------------------------------------------
    # Check that document exists
    # --------------------------------------------------------

    if not os.path.exists(document_path):

        print(
            f"WARNING: Document not found: "
            f"{document_path}"
        )

        continue


    # --------------------------------------------------------
    # 1. Ingest / Parse document
    # --------------------------------------------------------

    print("  → Ingesting document...")

    json_result = ingest_pdf(
        document_path
    )


    # --------------------------------------------------------
    # 2. Build documents
    # --------------------------------------------------------

    print("  → Building document nodes...")

    markdown_nodes = build_documents(
        json_result,
        document,
    )


    # --------------------------------------------------------
    # 3. Split into chunks
    # --------------------------------------------------------

    print("  → Splitting document into chunks...")

    final_chunks = split_markdown_document(
        markdown_nodes
    )


    print(
        f"  → Created {len(final_chunks)} chunks."
    )


    # --------------------------------------------------------
    # 4. Upsert chunks into vector database
    # --------------------------------------------------------

    print(
        "  → Upserting chunks into vector database..."
    )

    upsert_split_documents(
        final_chunks,
        USER_ID,
        document,
    )


    print(
        f"  ✓ Finished indexing: {document}"
    )


print("\n" + "=" * 60)
print("DOCUMENT INDEXING COMPLETE")
print("=" * 60)


# ============================================================
# 8. EVALUATE QUESTIONS
# ============================================================

evaluating_documents = []


print("\n" + "=" * 60)
print("STARTING RAG EVALUATION")
print("=" * 60)


for doc in tqdm(
    ground_truth,
    desc="Evaluating questions",
):

    question = doc["question"]

    source_document = doc["source"]

    reference_answer = doc["ground_truth"]


    # ========================================================
    # Validate source document
    # ========================================================

    if source_document not in documents:

        print(
            f"\nWARNING: Skipping question."
        )

        print(
            f"Source document '{source_document}' "
            f"is not in the evaluation document list."
        )

        continue


    print(
        f"\n{'-' * 60}"
    )

    print(
        f"Question: {question}"
    )

    print(
        f"Source: {source_document}"
    )


    # ========================================================
    # 1. RETRIEVE
    # ========================================================

    retrieved_documents = retrieve_context(
        query=question,
        user_id=USER_ID,
        source_document=source_document,
        top_k=5,
    )


    print(
        f"Retrieved documents: "
        f"{len(retrieved_documents)}"
    )


    # ========================================================
    # 2. RERANK
    # ========================================================

    if retrieved_documents:

        reranked_documents = rerank_results(
            query=question,
            documents=retrieved_documents,
            top_n=3,
        )

    else:

        print(
            "WARNING: No documents retrieved."
        )

        reranked_documents = []


    # ========================================================
    # 3. EXTRACT CONTEXT
    # ========================================================

    reranked_contexts = [
        document.page_content
        for document in reranked_documents
    ]


    # ========================================================
    # 4. BUILD CONTEXT STRING
    # ========================================================

    context_text = "\n\n".join(
        reranked_contexts
    )


    if not context_text:

        context_text = (
            "No relevant context was retrieved "
            "from the knowledge base."
        )


    # ========================================================
    # 5. FORMAT SYSTEM PROMPT
    # ========================================================

    formatted_system_prompt = system_prompt.format(
        contexts=context_text,
        query=question,
    )


    # ========================================================
    # 6. CREATE LLM MESSAGES
    # ========================================================

    system_message = SystemMessage(
        content=formatted_system_prompt
    )

    human_message = HumanMessage(
        content=question
    )


    messages = [
        system_message,
        human_message,
    ]


    # ========================================================
    # 7. GENERATE ANSWER
    # ========================================================

    response = llm.invoke(
        messages
    )

    answer = response.content


    # ========================================================
    # 8. SAVE EVALUATION RESULT
    # ========================================================

    evaluating_documents.append(
        {
            "question": question,

            "contexts": reranked_contexts,

            "answer": answer,

            "ground_truth": reference_answer,
        }
    )


# ============================================================
# 9. SAVE RAGAS EVALUATION DATASET
# ============================================================

print("\n" + "=" * 60)
print("SAVING EVALUATION DATASET")
print("=" * 60)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        evaluating_documents,
        f,
        indent=4,
        ensure_ascii=False,
    )


print(
    f"\n✓ Evaluation completed."
)

print(
    f"✓ Evaluated questions: "
    f"{len(evaluating_documents)}"
)

print(
    f"✓ Results saved to: "
    f"{OUTPUT_FILE}"
)