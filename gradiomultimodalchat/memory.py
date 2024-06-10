import ollama
import chromadb
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGMemory:
    def __init__(self, model="mxbai-embed-large", collection_name="docs", history_file="history.json", max_tokens=2048):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)
        self.history = []
        self.model = model
        self.history_file = history_file
        self.max_tokens = max_tokens
        self.load_history()

    def add_documents(self, documents):
        for i, doc in enumerate(documents):
            response = ollama.embeddings(model=self.model, prompt=doc)
            embedding = response["embedding"]
            self.collection.add(ids=[str(i)], embeddings=[embedding], documents=[doc])

    def retrieve_document(self, prompt, max_results=1):
        response = ollama.embeddings(model=self.model, prompt=prompt)
        query_embedding = response["embedding"]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=max_results)
        print("DEBUG: Query results:", results)  # Debug statement to inspect the results

        # Check if the results are not empty and contain valid documents
        if results and 'documents' in results and results['documents'] and results['documents'][0]:
            return results['documents'][0]
        else:
            print("DEBUG: No relevant documents found.")
            return None

    def add_to_history(self, user_input, response):
        self.history.append({'user_input': user_input, 'response': response})
        self.save_history()

    def get_history(self, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.max_tokens
        total_tokens = 0
        history_chunks = []
        for entry in reversed(self.history):
            user_input_tokens = len(entry['user_input'].split())
            response_tokens = len(entry['response'].split())
            if total_tokens + user_input_tokens + response_tokens > max_tokens:
                break
            history_chunks.append(entry)
            total_tokens += user_input_tokens + response_tokens
        history_chunks.reverse()
        return history_chunks

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f)

    def load_history(self):
        try:
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        except FileNotFoundError:
            self.history = []

    def clear_history(self):
        self.history = []
        self.save_history()

    def get_embedding(self, text):
        response = ollama.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def retrieve_similar_contexts(self, prompt_embedding, max_tokens=4096):
        results = self.collection.query(query_embeddings=[prompt_embedding], n_results=10)
        if not results or not results['documents']:
            return []
        contexts = []
        total_tokens = 0
        for doc_list in results['documents']:
            if not doc_list:
                continue
            doc = doc_list[0]
            doc_tokens = len(doc.split())
            if total_tokens + doc_tokens > max_tokens:
                break
            contexts.append(doc)
            total_tokens += doc_tokens
        return contexts
