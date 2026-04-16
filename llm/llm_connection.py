"""
llm_connection.py — Groq LLM wrapper with conversation history support.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt"
with open(_PROMPT_FILE, "r") as f:
    _system_prompt = f.read()


class LLM:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
        )

    def generate_response(
        self,
        query: str,
        context: list,
        history: list = None,
        system_prompt: str = _system_prompt,
    ) -> str:
        """
        Args:
            query:         the user's question
            context:       list of chunk dicts from VectorDB.search()
            history:       list of {"role": ..., "content": ...} for prior turns
            system_prompt: override the default system prompt
        """
        # Build context block from retrieved chunks
        context_text = "\n\n---\n\n".join(
            f"[Page {c.get('page', '?')}] {c['content']}" for c in context
        )

        # Assemble messages: system → history → new user turn
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext from document:\n{context_text}",
            }
        )

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
        )
        return response.choices[0].message.content
