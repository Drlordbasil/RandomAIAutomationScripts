import requests

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

def transcribe_audio(query_str):
    """
    Transcribes the audio file specified by the query string using the Whisper API.

    This amazing function takes a query string representing the path to an audio file and sends it to the Whisper API for transcription.
    It handles potential errors gracefully and provides informative feedback to the user.

    The Whisper API is a powerful tool for converting speech to text, and this function leverages its capabilities to deliver accurate transcriptions.
    It reads the audio file, sends a POST request to the API, and retrieves the transcribed text from the JSON response.

    Args:
        query_str (str): The query string containing the path to the audio file. This should be a valid path to an existing audio file.

    Returns:
        str: The transcribed text from the audio file. If the transcription is successful, the function returns the text as a string.
             If an error occurs during the process, it returns None.

    Raises:
        FileNotFoundError: If the specified audio file is not found. This exception is caught and handled within the function.
        requests.exceptions.RequestException: If there is an error with the API request, such as an invalid API key or network issues.
                                              This exception is caught and handled within the function.

    Example:
        >>> query_str = "path/to/audio/file.flac"
        >>> transcribed_text = transcribe_audio(query_str)
        >>> if transcribed_text:
        ...     print(transcribed_text)
        ... else:
        ...     print("Transcription failed.")

    Note:
        Make sure to replace the placeholder API key in the 'headers' variable with your actual Hugging Face API key.
        The audio file should be in a supported format, such as FLAC or WAV.

    Happy transcribing!
    """
    try:
        print(f"[DEBUG] Opening audio file: {query_str}")
        with open(query_str, "rb") as f:
            data = f.read()
        print("[DEBUG] Audio file read successfully")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[USER FEEDBACK] Oops! The specified audio file could not be found. Please check the file path and try again.")
        return None

    try:
        print("[DEBUG] Sending API request...")
        response = requests.post(API_URL, headers=headers, data=data)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        print("[DEBUG] API request successful")
        transcribed_text = response.json()['text']
        print("[DEBUG] Transcription received")
        return transcribed_text
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {e}")
        print("[USER FEEDBACK] Uh-oh! There was an error with the API request. Please check your API key and network connection.")
        return None

if __name__ == "__main__":
    query_str = "sample2.flac"
    print(f"[USER FEEDBACK] Transcribing audio file: {query_str}")
    output = transcribe_audio(query_str)
    if output:
        print("[USER FEEDBACK] Transcription successful!")
        print("[TRANSCRIBED TEXT]")
        print(output)
    else:
        print("[USER FEEDBACK] Transcription failed. Please check the error messages above for more details.")
