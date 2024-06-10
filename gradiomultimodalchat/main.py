from memory import RAGMemory
from generate_response import ResponseGenerator
from vision import VisionAgent
from speech_recognition import SpeechRecognition
from text_to_speech import TextToSpeech
import gradio as gr
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
# Initialize components
memory = RAGMemory()
response_generator = ResponseGenerator()
vision_agent = VisionAgent()
speech_recognition = SpeechRecognition()
text_to_speech = TextToSpeech()

# System message
system_message = "Welcome to the RAG-based chat interface. Ask me anything!"

def chat_with_agent(message, history, image=None, audio=None):
    if image is not None:
        vision_description = vision_agent.describe_image(image, memory)
        memory.add_to_history("image input", vision_description)
        return history + [(message, vision_description)]
    elif audio is not None:
        audio_transcription = speech_recognition.query(audio)
        if "error" in audio_transcription:
            return history + [(message, audio_transcription["error"])]
        message = audio_transcription.get('text', 'Unable to transcribe audio.')
        response = response_generator.generate_response(message, memory)
        memory.add_to_history(message, response)
        return history + [(message, response)]
    elif message:
        response = response_generator.generate_response(message, memory)
        memory.add_to_history(message, response)
        
        # Generate speech from the response text
        audio_bytes = text_to_speech.query({"inputs": response})
        if audio_bytes:
            text_to_speech.save_audio_to_file(audio_bytes, "response_audio.wav")
            return history + [(message, response), (None, "response_audio.wav")]
        return history + [(message, response)]
    else:
        return history + [(message, "Please provide either a text input, an image, or an audio file.")]

def get_chat_history():
    history = memory.get_history()
    formatted_history = "\n".join([f"User: {entry['user_input']}\nResponse: {entry['response']}" for entry in history])
    return formatted_history

def run_gradio_interface():
    # Define Gradio interface
    with gr.Blocks() as interface:
        gr.Markdown(system_message)
        
        chatbot = gr.Chatbot(label="Chatbot")
        
        with gr.Row():
            with gr.Column(scale=12):
                user_input = gr.Textbox(lines=10, placeholder="Type a message...", label="Message")
            with gr.Column(scale=3):
                image_input = gr.Image(type="filepath", label="Upload an image (optional)")
            with gr.Column(scale=3):
                audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload an audio file (optional)")
            with gr.Column(scale=1):
                submit_button = gr.Button("Submit")
        
        # Buttons for Retry, Undo, Clear
        with gr.Row():
            retry_button = gr.Button("Retry")
            undo_button = gr.Button("Undo")
            clear_button = gr.Button("Clear")
        
        def on_submit(message, history, image, audio):
            updated_history = chat_with_agent(message, history, image, audio)
            return updated_history, updated_history
        
        submit_button.click(on_submit, inputs=[user_input, chatbot, image_input, audio_input], outputs=[chatbot, chatbot])
        
        def on_retry(history):
            if history:
                last_message, last_response = history[-1]
                updated_history = chat_with_agent(last_message, history[:-1])
                return updated_history
            return history
        
        retry_button.click(on_retry, inputs=[chatbot], outputs=[chatbot])
        
        def on_undo(history):
            if history:
                updated_history = history[:-1]
                memory.undo_last_entry()
                return updated_history
            return history
        
        undo_button.click(on_undo, inputs=[chatbot], outputs=[chatbot])
        
        def on_clear():
            memory.clear_history()
            return []
        
        clear_button.click(on_clear, outputs=[chatbot])
    
    interface.launch()

if __name__ == "__main__":
    run_gradio_interface()
