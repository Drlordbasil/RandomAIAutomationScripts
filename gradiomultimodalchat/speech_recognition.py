import requests
import time

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

class SpeechRecognition:
    def __init__(self, api_url=API_URL, headers=headers, max_retries=3, retry_delay=5):
        self.api_url = api_url
        self.headers = headers
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def query(self, filename):
        retries = 0
        while retries < self.max_retries:
            try:
                with open(filename, "rb") as f:
                    data = f.read()
                response = requests.post(self.api_url, headers=self.headers, data=data)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error querying the speech recognition API: {e}")
                if "503" in str(e):
                    retries += 1
                    print(f"Retrying... ({retries}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    return {"error": str(e)}
            except Exception as e:
                print(f"Unexpected error: {e}")
                return {"error": str(e)}
        return {"error": "Max retries exceeded. Please try again later."}

# Example usage
#if __name__ == "__main__":
    #recognizer = SpeechRecognition()
    #output = recognizer.query("answer_to_universe.wav")
    #print(output)
