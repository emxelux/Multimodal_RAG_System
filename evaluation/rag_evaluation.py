# from ragas import evaluate
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
# )
# from datasets import Dataset
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from tqdm.auto import tqdm
from data_preprocessing.vector_db import get_vector_store, rerank_results, upsert_split_documents, retrieve_context
from qdrant_client import models
from tqdm.auto import tqdm
from data_preprocessing.ingest import ingest_pdf, build_documents
from data_preprocessing.chunking import split_markdown_document
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

import json

with open("evaluation/Documents/ground_truth.json") as f:
    ground_truth = json.load(f)

with open("prompts/system_prompt_v2.txt") as f:
    system_prompt_template = f.read()

llm = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq"
)


def evaluation_and_testing(file_path:str):
    json_result = ingest_pdf(file_path)
    document = build_documents(json_result, source_name=file_path)
    chunks = split_markdown_document(document)
    return chunks

test_docs = ["Technical Paper on Machine Learning.pdf", "DATA-ANALYSIS-REPORT-TEAM-5.pdf", "Medical Paper in Cancer.pdf"]

for doc in test_docs:
    for que in tqdm(ground_truth):
        if que["source"]+".pdf" == doc:
            chunks = evaluation_and_testing(f"evaluation/Documents/{doc}")
            upsert_split_documents(chunks, user_id="ae04cd70-1b45-43ee-9666-b83602180bdd", source_document=doc)
            context = retrieve_context(que["question"], user_id="ae04cd70-1b45-43ee-9666-b83602180bdd", source_document=doc, top_k=5)
            reranked_context = rerank_results(que["question"], context, top_n=3)
            system_prompt_template = system_prompt_template.format(query=que["question"], doc_context=reranked_context).strip()
            system_message = SystemMessage(content=system_prompt_template)
            human_message = HumanMessage(content=que["question"])
            messages = [system_message, human_message]
            answer = llm.invoke(messages)
            que["answer"] = answer.content
            que["context"] = reranked_context
with open("ground_truth_with_answers.json", "w") as f:
    json.dump(ground_truth, f, indent=4)