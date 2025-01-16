from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path("../.env")
load_dotenv(dotenv_path=env_path)

OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
  api_key=OPEN_AI_KEY
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message)