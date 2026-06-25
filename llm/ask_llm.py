from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()

with open("prompts/system_prompt.txt") as f_read:
    system_prompt = f_read.read()


os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


model = init_chat_model(
    model = "gemini-3.5-flash",
    model_provider="google_genai"
)


# Create a template combining System and Human messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the prompt \n{system_prompt} to answer {user_question} from {context}"),
    ("human", "{user_question}")
])


def generation(query, doc_context):
    formatted_messages = prompt.format_messages(
    system_prompt = system_prompt,
    user_question = query,
    context = doc_context
)
    response = model.invoke(formatted_messages)

    return response.content
