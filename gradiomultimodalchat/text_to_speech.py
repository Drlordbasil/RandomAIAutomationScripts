import requests
from IPython.display import Audio, display
import IPython.display as ipd

API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
headers = {"Authorization": "Bearer hf_"}

class TextToSpeech:
    def __init__(self, api_url=API_URL, headers=headers):
        self.api_url = api_url
        self.headers = headers

    def query(self, payload):
        """
        Sends a text-to-speech query to the API.
        
        Args:
            payload (dict): The payload containing the input text.

        Returns:
            bytes: The audio content returned by the API.
        """
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return None

    def save_audio_to_file(self, audio_bytes, filename):
        """
        Saves the audio bytes to a file.
        
        Args:
            audio_bytes (bytes): The audio content to save.
            filename (str): The filename to save the audio content as.
        """
        try:
            with open(filename, "wb") as f:
                f.write(audio_bytes)
            print(f"Audio saved to {filename}")
        except Exception as e:
            print(f"Error saving audio to file: {e}")

if __name__ == "__main__":
    tts = TextToSpeech()
    text_input = "hello my love I am here and queer."
    audio_bytes = tts.query({"inputs": text_input})

    if audio_bytes:
        # Display audio in notebook
        display(Audio(audio_bytes))

        # Save the audio to a file
        filename = "answer_to_universe.wav"
        tts.save_audio_to_file(audio_bytes, filename)

        # Play the audio from the file
        display(ipd.Audio(filename))
    else:  
        print("Failed to retrieve audio from the API.")
