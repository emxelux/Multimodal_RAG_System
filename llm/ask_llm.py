from pathlib import Path
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


def stream_generation(query: str, doc_context: str):
    """
    Generator that yields raw text tokens from the LLM one chunk at a time.
    Call this inside a FastAPI StreamingResponse.
    """
    model = init_chat_model(
        model="openai/gpt-oss-120b",
        model_provider="groq"
    )

    system_prompt_template = Path("prompts/system_prompt.txt").read_text()
    system_prompt = system_prompt_template.format(query=query, doc_context=doc_context)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_question}"),
    ])

    formatted = prompt.format_messages(
        user_question=query,
    )
    logger.info(f"========================\n FORMATTED PROMPT TO BE SENT TO LLM: \n {formatted}\n ===============")

    for chunk in model.stream(formatted):
        if chunk.content:
            text = chunk.content
            if isinstance(text, list):
                text = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in text
                )
            if text:
                yield text


def generation(query: str, doc_context: str):
    """Return the full LLM response as a single string."""
    return "".join(stream_generation(query=query, doc_context=doc_context))