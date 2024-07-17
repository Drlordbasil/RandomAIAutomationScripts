import secrets
import string
import logging
import math
import sqlite3
import time
import tensorflow as tf
import numpy as np
from openai import OpenAI, OpenAIError

# Provided keys as training references
reference_keys = [ # all keys are not real, dw.
    'sk-proj-jXwmFRD6pFJVCBCMPIV1',
    'sk-proj-',
    ''
]

DATABASE = "keys.db"
RATE_LIMIT = 60  # seconds
MAX_RETRIES = 5
DELAY_BETWEEN_KEYS = 2  # seconds

def initialize_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tried_keys (
            key TEXT PRIMARY KEY
        )
    """)
    conn.commit()
    conn.close()

def has_key_been_tried(key):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM tried_keys WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def save_tried_key(key):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tried_keys (key) VALUES (?)", (key,))
    conn.commit()
    conn.close()

def generate_sk_proj_series(length=48):
    prefix = 'sk-proj-'
    total_length = length - len(prefix)
    target_counts = {'lowercase': 20, 'uppercase': 20, 'digits': 8}
    
    characters = [secrets.choice(string.ascii_lowercase) for _ in range(target_counts['lowercase'])]
    characters += [secrets.choice(string.ascii_uppercase) for _ in range(target_counts['uppercase'])]
    characters += [secrets.choice(string.digits) for _ in range(target_counts['digits'])]
    
    secrets.SystemRandom().shuffle(characters)
    
    return prefix + ''.join(characters)

def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

def create_completion(client):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
            ]
        )
        return completion
    except OpenAIError as e:
        logging.error(f"OpenAI API request failed: {e}")
        return None

def save_key_to_file(api_key, filename="keys.txt"):
    with open(filename, 'a') as file:
        file.write(api_key + '\n')
    logging.info(f"Saved valid API key to {filename}")

class KeyGeneratorModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=(48,))
        dense_1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        dense_2 = tf.keras.layers.Dense(64, activation='relu')(dense_1)
        output_layer = tf.keras.layers.Dense(48, activation='sigmoid')(dense_2)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, key_vector):
        return self.model.predict(np.array([key_vector]))

    def train(self, input_vectors, target_vectors):
        self.model.fit(np.array(input_vectors), np.array(target_vectors), epochs=10)

def key_to_vector(key):
    char_to_int = {c: i for i, c in enumerate(string.ascii_letters + string.digits)}
    return [char_to_int[char] for char in key]

def vector_to_key(vector):
    int_to_char = string.ascii_letters + string.digits
    return ''.join(int_to_char[int(round(v))] for v in vector)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    initialize_database()

    key_generator_model = KeyGeneratorModel()

    input_vectors = [key_to_vector(key[len('sk-proj-'):]) for key in reference_keys]
    target_vectors = input_vectors.copy()

    key_generator_model.train(input_vectors, target_vectors)

    retries = 0

    while True:
        generated_series = generate_sk_proj_series()
        while has_key_been_tried(generated_series):
            generated_series = generate_sk_proj_series()

        save_tried_key(generated_series)
        logging.info(f"Generated API Key: {generated_series}")

        key_vector = key_to_vector(generated_series[len('sk-proj-'):])

        improved_vector = key_generator_model.predict(key_vector)
        improved_key = 'sk-proj-' + vector_to_key(improved_vector[0])

        if has_key_been_tried(improved_key) or improved_key in reference_keys:
            continue

        save_tried_key(improved_key)

        client = initialize_openai_client(api_key=improved_key)

        for attempt in range(MAX_RETRIES):
            try:
                completion = create_completion(client)
                if completion:
                    print(completion.choices[0].message)
                    save_key_to_file(improved_key)
                    retries = 0
                    break
            except OpenAIError as e:
                logging.error(f"API request failed: {e}. Retrying...")
                time.sleep(min(RATE_LIMIT, 2 ** attempt))
        else:
            logging.error("Failed to retrieve completion from OpenAI after several attempts, generating a new key.")
            key_generator_model.train([key_vector], [key_vector])
            retries += 1
            time.sleep(DELAY_BETWEEN_KEYS)

if __name__ == "__main__":
    main()
