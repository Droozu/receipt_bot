import base64
import requests
import json
import csv
import os
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")


API_URL = "https://api.deepseek.com"
client = OpenAI(
    api_key="sk-f5d743133bda4b12ab55773c07d6c629",
    base_url=API_URL
    )


# ----------------------------
# Encode image to Base64
# ----------------------------
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ----------------------------
# Send receipt to DeepSeek
# ----------------------------
def parse_receipt(image_path: str):

    img_b64 = encode_image(image_path)
    print(img_b64)

    prompt = """
You are a receipt parser.

Extract structured data from the receipt image.

Return ONLY valid JSON:

{
 "store_name": "",
 "legal_name": "",
 "datetime": "",
 "items":[
    {"name":"", "quantity":0, "price":0}
 ]
}
"""

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=payload["messages"],
        response_format={
        'type': 'json_object'
        }
        )



    return json.loads(response.choices[0].message.content)


# ----------------------------
# Save JSON
# ----------------------------
def save_json(data, path="receipt.json"):

    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ----------------------------
# Save CSV
# ----------------------------
def save_csv(data, path="items.csv"):

    with open(path, "w", newline="", encoding="utf8") as f:

        writer = csv.writer(f)
        writer.writerow(["name", "quantity", "price"])

        for item in data["items"]:
            writer.writerow([
                item["name"],
                item["quantity"],
                item["price"]
            ])


# ----------------------------
# Main
# ----------------------------
def run(image_path):

    result = parse_receipt(image_path)

    save_json(result)
    save_csv(result)

    print("Parsed receipt:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python receipt_client.py receipt.jpg")
        exit()

    run(sys.argv[1])