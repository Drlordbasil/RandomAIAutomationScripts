import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from datasets import load_dataset

nltk.download('punkt')
nltk.download('vader_lexicon')

class DataPreprocessor:
    def __init__(self, dataset_name, split='train'):
        print("Loading dataset...")
        self.dataset = load_dataset(dataset_name, split=split)
        print(f"Dataset loaded: {len(self.dataset)} samples")
        self.vocab = set()
        self.word_to_index = {}
        self.index_to_word = {}
        self.max_seq_length = 0

    def tokenize(self, text):
        print(f"Tokenizing text: {text}")
        return word_tokenize(text.lower())

    def preprocess_data(self):
        print("Preprocessing data...")
        user_inputs = []
        assistant_responses = []
        for item in self.dataset:
            try:
                print(f"Processing item: {item}")
                user_tokens = self.tokenize(item['prompt'])
                assistant_tokens_1 = self.tokenize(item['response_A'])
                assistant_tokens_2 = self.tokenize(item['response_B'])
                print(f"User tokens: {user_tokens}")
                print(f"Assistant tokens 1: {assistant_tokens_1}")
                print(f"Assistant tokens 2: {assistant_tokens_2}")
                user_inputs.append(user_tokens)
                assistant_responses.append(assistant_tokens_1)
                user_inputs.append(user_tokens)
                assistant_responses.append(assistant_tokens_2)
            except KeyError as e:
                print(f"KeyError: {e} in item: {item}")
        print(f"Total user inputs: {len(user_inputs)}")
        print(f"Total assistant responses: {len(assistant_responses)}")
        print("Preprocessing complete.")
        return user_inputs, assistant_responses

    def build_vocabulary(self, user_inputs, assistant_responses):
        print("Building vocabulary...")
        if not user_inputs and not assistant_responses:
            print("Error: No data to build vocabulary from.")
            raise ValueError("No data to build vocabulary from.")
        
        self.vocab = set(word for sentence in user_inputs + assistant_responses for word in sentence)
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab, 1)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        self.max_seq_length = max(len(seq) for seq in user_inputs + assistant_responses)
        print(f"Vocabulary built with {len(self.vocab)} words.")
        print(f"Max sequence length: {self.max_seq_length}")

    def texts_to_sequences(self, texts):
        print("Converting texts to sequences...")
        sequences = [[self.word_to_index[word] for word in text if word in self.word_to_index] for text in texts]
        print(f"Converted sequences: {sequences}")
        return sequences

    def pad_sequences(self, sequences, dim=4):
        print("Padding sequences...")
        padded_sequences = np.array([seq + [0] * (self.max_seq_length - len(seq)) for seq in sequences]).reshape(-1, dim, self.max_seq_length // dim)
        print(f"Padded sequences: {padded_sequences}")
        return padded_sequences

    def prepare_data(self):
        print("Preparing data...")
        user_inputs, assistant_responses = self.preprocess_data()
        print("Building vocabulary...")
        self.build_vocabulary(user_inputs, assistant_responses)
        print("Converting user inputs to sequences...")
        user_sequences = self.texts_to_sequences(user_inputs)
        print("Converting assistant responses to sequences...")
        assistant_sequences = self.texts_to_sequences(assistant_responses)
        print("Padding user sequences...")
        user_sequences = self.pad_sequences(user_sequences)
        print("Padding assistant sequences...")
        assistant_sequences = self.pad_sequences(assistant_sequences)
        Y_train = np.array([seq[0][0] for seq in assistant_sequences])
        print("Data preparation complete.")
        return user_sequences, assistant_sequences, Y_train

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100, num_layers=2):
        super(LSTMModel, self).__init__()
        print("Initializing LSTMModel...")
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        print("LSTMModel initialized.")

    def forward(self, x):
        print(f"Forward pass with input: {x}")
        x = self.embedding(x)
        h_0 = torch.zeros(2, x.size(0), 100).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 100).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        print(f"Forward pass output: {out}")
        return out

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        print("Trainer initialized.")

    def train(self, X_train, Y_train, num_epochs=1000):
        print("Starting training...")
        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            self.optimizer.zero_grad()
            print(f"Epoch {epoch+1}/{num_epochs}")
            outputs = self.model(X_train)
            print(f"Model outputs: {outputs}")
            loss = self.criterion(outputs, Y_train)
            print(f"Loss: {loss.item()}")
            loss.backward()

            # Apply gradient vectorization
            with torch.no_grad():
                for param in self.model.parameters():
                    param.grad = param.grad / (torch.norm(param.grad, p=4) + 1e-8)

            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print("Training complete.")

class Chatbot:
    def __init__(self, model, word_to_index, index_to_word, max_seq_length):
        self.model = model
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.max_seq_length = max_seq_length
        self.sia = SentimentIntensityAnalyzer()
        print("Chatbot initialized.")

    def generate_response(self, input_text):
        print(f"Generating response for: {input_text}")
        input_tokens = word_tokenize(input_text.lower())
        print(f"Input tokens: {input_tokens}")

        # Update vocabulary with new words
        for word in input_tokens:
            if word not in self.word_to_index:
                new_index = len(self.word_to_index) + 1
                self.word_to_index[word] = new_index
                self.index_to_word[new_index] = word
                print(f"New word added to vocabulary: {word} -> {new_index}")

        input_sequence = [[self.word_to_index[word] for word in input_tokens if word in self.word_to_index]]
        print(f"Input sequence: {input_sequence}")
        input_sequence = np.array([seq + [0] * (self.max_seq_length - len(seq)) for seq in input_sequence]).reshape(-1, 4, self.max_seq_length // 4)
        print(f"Padded input sequence: {input_sequence}")
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        print(f"Input tensor: {input_tensor}")

        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output_prob = self.model(input_tensor)
        print(f"Output probabilities: {output_prob}")
        output_sequence = torch.argmax(output_prob, dim=1).numpy()
        print(f"Output sequence: {output_sequence}")

        response = " ".join(self.index_to_word[idx] for idx in output_sequence if idx > 0)
        sentiment = self.sia.polarity_scores(input_text)

        print(f"Response: {response}")
        print(f"Sentiment: {sentiment}")
        return response, sentiment

if __name__ == "__main__":
    # Preprocess data
    print("Initializing DataPreprocessor...")
    preprocessor = DataPreprocessor('lmsys/lmsys-arena-human-preference-55k')
    user_sequences, assistant_sequences, Y_train = preprocessor.prepare_data()

    # Prepare data for training
    print("Preparing training data...")
    vocab_size = len(preprocessor.vocab) + 1
    X_train = torch.tensor(user_sequences, dtype=torch.long)
    Y_train = torch.tensor(Y_train, dtype=torch.long)

    # Initialize and train the model
    print("Initializing LSTMModel...")
    model = LSTMModel(input_size=vocab_size, output_size=vocab_size, hidden_size=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    print("Initializing Trainer...")
    trainer = Trainer(model, criterion, optimizer)
    trainer.train(X_train, Y_train)

    # Generate response
    print("Initializing Chatbot...")
    chatbot = Chatbot(model, preprocessor.word_to_index, preprocessor.index_to_word, preprocessor.max_seq_length)
    input_text = "Hey can you tell me about cookies?"
    response, sentiment = chatbot.generate_response(input_text)
    print(f"Input: {input_text}")
    print(f"Output: {response}")
    print(f"Sentiment: {sentiment}")a
