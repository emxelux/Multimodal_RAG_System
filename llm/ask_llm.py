from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

def stream_generation(query: str, doc_context: str):
    """
    Generator that yields raw text tokens from the LLM one chunk at a time.
    Call this inside a FastAPI StreamingResponse.
    """
    model = init_chat_model(
        model="gemini-2.0-flash",
        model_provider="google_genai"
    )

    system_prompt = Path("prompts/system_prompt.txt").read_text()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_question}")
    ])

    formatted = prompt.format_messages(
        system_prompt=system_prompt,
        user_question=query,
        context=doc_context
    )

    # model.stream() yields AIMessageChunk objects instead of one AIMessage
    for chunk in model.stream(formatted):
        if chunk.content:
            # content can be str or list — normalise to str
            text = chunk.content
            if isinstance(text, list):
                text = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in text
                )
            if text:
                yield text