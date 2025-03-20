import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0,max_tokens=100)

messages = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "I love programming and Eden Hazard"}
]


ai_msg = llm.invoke(messages)
print(ai_msg)