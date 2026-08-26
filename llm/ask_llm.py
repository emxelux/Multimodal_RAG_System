from pathlib import Path
from typing import Optional
import logging
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)


def build_history_messages(history: Optional[list] = None):
    """Convert prior turns into LangChain message objects for the prompt."""
    history = history or []
    messages = []

    for turn in history:

        if isinstance(turn, (list, tuple)):
            if len(turn) < 2:
                continue
            role = str(turn[0]).lower()
            content = turn[1]
            

        elif isinstance(turn, dict):
            role = str(turn.get("role", "")).lower()
            content = turn.get("content") or turn.get("text") or ""
            
        else:

            continue

        if not content:
            continue


        if role in {"assistant", "ai", "model"}:
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    return messages


def stream_generation(query: str, doc_context: str, history: Optional[list[dict]] = None):
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
    
    history_messages = build_history_messages(history)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_question}"),
    ])

    formatted = prompt.format_messages(
        history=history_messages,
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


def generation(query: str, doc_context: str, history: Optional[list[dict]] = None):
    """Return the full LLM response as a single string."""
    return "".join(stream_generation(query=query, doc_context=doc_context, history=history))