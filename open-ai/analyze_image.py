import base64
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

def decode_image(image_path):
  image_file = open(image_path, "rb")
  return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "/Users/sahilsaini/Desktop/Projects/sign-to-text/hand_detection/output_data/test-image-2.png"
base64_image = decode_image(image_path)

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text", 
          "text": "do the hand symbol have any ASL meaning in this image? if you think it has a valid meaning then response with only the ASL translation, otherwise return none."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
          }
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0].message.content)