import ollama
from memory import RAGMemory

class VisionAgent:
    def __init__(self, model="llava-llama3"):
        self.model = model
        print(f"[INFO] Initialized VisionAgent with model {model}")

    def describe_image(self, image_path, memory: RAGMemory):
        print(f"[INFO] Describing image: {image_path}")
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            print("[INFO] Sending image to LLM for description")
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image:',
                    'images': [image_data]
                }]
            )
            
            description = response['message']['content']
            print(f"[INFO] Received image description: {description[:50]}...")  # Print first 50 characters
            memory.add_to_history("image description", description)
            
            return description
        except Exception as e:
            error_message = f"[ERROR] An error occurred while describing the image: {str(e)}"
            print(error_message)
            return error_message
