import hashlib
import re
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Ensure necessary NLTK data is downloaded
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TokenManager:
    def __init__(self):
        self.base_mapping = {
            "the": "α", "and": "β", "ing": "γ", ".": "·", ",": "‚", "!": "ǃ", "?": "¿"
        }
        self.token_mapping = self.base_mapping.copy()
        self.reverse_token_mapping = {v: k for k, v in self.base_mapping.items()}
    
    def generate_symbol(self, word):
        """Generates a consistent symbol for a word using a hash function."""
        # Use a hashing function to generate a unique but consistent identifier for each word
        word_hash = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        return f"τ{word_hash}"

    def update_mappings(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        for word in set(words):
            if word not in self.token_mapping and word not in self.base_mapping:
                symbol = self.generate_symbol(word)
                self.token_mapping[word] = symbol
                self.reverse_token_mapping[symbol] = word

class Encoder:
    def __init__(self, token_manager):
        self.token_manager = token_manager

    def encode(self, text):
        for word, symbol in self.token_manager.token_mapping.items():
            text = re.sub(r'\b' + re.escape(word) + r'\b', symbol, text)
        return text

class Decoder:
    def __init__(self, token_manager):
        self.token_manager = token_manager

    def decode(self, encoded_text):
        for symbol, word in self.token_manager.reverse_token_mapping.items():
            encoded_text = encoded_text.replace(symbol, word)
        return encoded_text

class CompressoLangTranslator:
    def __init__(self):
        self.token_manager = TokenManager()
        self.encoder = Encoder(self.token_manager)
        self.decoder = Decoder(self.token_manager)

    def encode(self, text):
        self.token_manager.update_mappings(text)
        return self.encoder.encode(text)

    def decode(self, text):
        return self.decoder.decode(text)

class Utility:
    @staticmethod
    def preprocess_text(text):
        text = text.lower().strip()
        return text

if __name__ == "__main__":
    translator = CompressoLangTranslator()
    original_text = "Sample text to encode and decode using CompressoLang."
    processed_text = Utility.preprocess_text(original_text)
    encoded_text = translator.encode(processed_text)
    decoded_text = translator.decode(encoded_text)

    print(f"Original: {original_text}")
    print(f"Processed: {processed_text}")
    print(f"Encoded: {encoded_text}")
    print(f"Decoded: {decoded_text}")
