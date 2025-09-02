# test_api.py
import requests
import json

url = "http://127.0.0.1:8000/predict"
data = {
    "species": "dog",
    "weightKg": 23,
    "age": 2,
    "neutered": "yes",
    "activity": "medium",
    "symptoms": "itchy paws",
    "conditions": "obesity",
    "allergies": ["wheat"]
}

print(f"👉 Sending request to {url}")
try:
    response = requests.post(url, json=data)
    # Try to parse JSON even if status != 200
    try:
        payload = response.json()
    except Exception:
        payload = response.text

    if response.status_code == 200:
        print("✅ API call success!")
        print(json.dumps(payload, indent=2))
    else:
        print(f"❌ Request failed: {response.status_code}")
        print("Server response:")
        print(payload)

except Exception as e:
    print("❌ Could not connect to API:", str(e))
