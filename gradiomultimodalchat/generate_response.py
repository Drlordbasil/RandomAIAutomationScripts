from langchain_community.llms import Ollama
from memory import RAGMemory

class ResponseGenerator:
    def __init__(self, model="dolphin-llama3"):
        self.llm = Ollama(model=model)

    def generate_response(self, prompt, memory: RAGMemory):
        prompt_embedding = memory.get_embedding(prompt)
        contexts = memory.retrieve_similar_contexts(prompt_embedding)
        
        # Integrate history into the context
        history = memory.get_history()
        history_context = " ".join([f"User: {entry['user_input']} Response: {entry['response']}" for entry in history])
        
        combined_prompt = history_context + " " + " ".join(contexts) + " " + prompt

        # Ensure combined prompt does not exceed token limit
        combined_prompt_tokens = combined_prompt.split()
        if len(combined_prompt_tokens) > memory.max_tokens:
            combined_prompt = " ".join(combined_prompt_tokens[-memory.max_tokens:])

        response = self.llm.invoke(combined_prompt)
        memory.add_to_history(prompt, response)
        return response
