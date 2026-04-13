from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

with open('prompts/system_prompt.txt', 'r') as f:
    system_prompt = f.read()




class LLM:
    def __init__(self, api_key=os.getenv("GROQ_API_KEY")):
        self.api_key = api_key
        self.client = OpenAI(
            base_url="https://api.groq.com/api/openai/v1",
            api_key=self.api_key
            )


    def generate_response(self, query, result,system_prompt=system_prompt):
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}\n\nDocument:\n{result}"}
                      ]
        )
        return response.choices[0].message.content