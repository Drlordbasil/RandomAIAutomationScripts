
import json
import time
import os
import re
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

class OpenAIAPI:
    def __init__(self, api_key):
        self.smart_model = "llama3-70b-8192"
        self.fast_model = "llama3-8b-8192"
        self.rating_model = "mixtral-8x7b-32768"
        self.base_url = "https://api.groq.com/openai/v1"
        self.api_key = api_key
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.local_model_name = "local_llm"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_file = "responses/dataset.json"

    def complete_task(self, task_description, num_iterations=20):
        """
        Generates responses for the given task description using the fast and smart models,
        rates the responses, and saves them to the dataset file.
        """
        for i in range(num_iterations):
            print(f"Iteration {i+1}/{num_iterations}")

            print("Generating fast response...")
            fast_response = self.generate_response(task_description, self.fast_model)
            print("Fast response:", fast_response)

            print("Generating smart response...")
            smart_response = self.generate_response(task_description, self.smart_model)
            print("Smart response:", smart_response)

            print("Rating responses...")
            rating_response = self.rate_responses(task_description, fast_response, smart_response)
            print("Rating response:", rating_response)

            highest_rated_response = self.get_highest_rated_response(rating_response)
        return highest_rated_response

    def generate_response(self, task_description, model):
        """
        Generates a response for the given task description using the specified model.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI that completes given tasks."
                    },
                    {
                        "role": "user",
                        "content": f"Please provide the code for the following task: {task_description}"
                    }
                ],
                max_tokens=4000,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None

    def rate_responses(self, task_description, fast_response, smart_response):
        """
        Rates the fast and smart responses for the given task description.
        """
        try:
            rating_response = self.client.chat.completions.create(
                model=self.rating_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are sent 2 different AI responses and are required to rate them from 1-10 with a clear quantitative score like '9/10' at the start of each evaluation. Explain the differences and why which model is better. Be very detailed in your analysis and provide a realistic 1-10 rating always. Be harsh. Be honest. Be fair."
                    },
                    {
                        "role": "user",
                        "content": f"""
                        User task given to the models: {task_description}
                        Fast model response Agent1: {fast_response}
                        Smart model response Agent2: {smart_response}
                        Rate both responses from 1-10 and explain the differences and why which model is better.
                        Deduct points for any placeholders or missing code.
                        Deduct points for any code that is not robust or not working.
                        Deduct points for any code that is not standalone.
                        Deduct points for any code that is not pythonic.
                        Deduct points for any code that is not efficient.
                        Deduct points for any code that is not secure.
                        Add points for any code that is well documented.
                        Add points for any code that is well structured.
                        Add points for any code that is well tested.
                        Add points for any code that is well optimized.
                        Add points for any code that is well formatted.
                        Add points for any code that has proper flow control.
                        Add points for any code that has proper error handling.
                        Add points for any code that has proper variable naming.
                        you are tasked with rating the responses from 1-10 and explaining the differences and why which model is better based on the given task.
                        you should simply format your response as such:
                        model name: rating/10 
                        model name: rating/10
                        
                        """
                    }
                ],
                max_tokens=4000,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return rating_response.choices[0].message.content
        except Exception as e:
            print(f"Error rating responses: {str(e)}")
            return None


    def get_highest_rated_response(self, rating_response):
        """
        Extracts the ratings from the rating response and returns the highest rated response.
        """
        try:
            lines = rating_response.split("\n")
            fast_rating = None
            smart_rating = None

            for line in lines:
                if "Agent1" in line:
                    match = re.search(r'(\d+)/10', line)
                    if match:
                        fast_rating = int(match.group(1))
                elif "Agent2" in line:
                    match = re.search(r'(\d+)/10', line)
                    if match:
                        smart_rating = int(match.group(1))

            if fast_rating is None or smart_rating is None:
                raise ValueError("Unable to extract ratings from the rating response.")

            if fast_rating > smart_rating:
                return "Fast Response"
            elif smart_rating > fast_rating:
                return "Smart Response"
            else:
                return "Both responses are equally rated."
        except Exception as e:
            print(f"Error getting highest rated response: {str(e)}")
            return None

    def train_local_model(self):
        """
        Trains the local model using the dataset file.
        """
        try:
            if os.path.exists(self.local_model_name):
                model = AutoModelForCausalLM.from_pretrained(self.local_model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained("gpt2")
                tokenizer = AutoTokenizer.from_pretrained("gpt2")

            dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=self.dataset_file,
                block_size=128,
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False,
            )

            training_args = TrainingArguments(
                output_dir="./results",
                overwrite_output_dir=True,
                num_train_epochs=2,
                per_device_train_batch_size=32,
                save_steps=10_000,
                save_total_limit=2,
                prediction_loss_only=True,
                learning_rate=5e-5,
                weight_decay=0.01,
                warmup_steps=500,
                logging_dir='./logs',
                logging_steps=100,
                no_cuda=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )

            trainer.train()

            os.makedirs(self.local_model_name, exist_ok=True)
            model.save_pretrained(self.local_model_name)
            tokenizer.save_pretrained(self.local_model_name)
        except Exception as e:
            print(f"Error training local model: {str(e)}")

    def generate_local_response(self, prompt):
        """
        Generates a response using the trained local model.
        """
        try:
            if not os.path.exists(self.local_model_name):
                raise ValueError("Local model not found. Please train the model first.")

            model = AutoModelForCausalLM.from_pretrained(self.local_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            output = model.generate(
                input_ids,
                max_length=3000,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

            response = tokenizer.decode(output[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"Error generating local response: {str(e)}")
            return None

    def evaluate_response(self, prompt, generated_response):
        """
        Evaluates the generated response by comparing it with the reference response.
        """
        try:
            reference_response = self.generate_response(prompt, self.smart_model)

            reference_words = reference_response.split()
            generated_words = generated_response.split()

            reference_counts = {}
            for word in reference_words:
                if word not in reference_counts:
                    reference_counts[word] = 0
                reference_counts[word] += 1

            overlap = 0
            for word in generated_words:
                if word in reference_counts and reference_counts[word] > 0:
                    reference_counts[word] -= 1
                    overlap += 1

            precision = overlap / len(generated_words)
            recall = overlap / len(reference_words)

            if precision + recall == 0:
                f1_score = 0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

            return f1_score
        except Exception as e:
            print(f"Error evaluating response: {str(e)}")
            return None

    def train_and_evaluate(self, task_description, prompt):
        """
        Trains the local model and evaluates the generated response.
        """
        num_iterations = 20
        sleep_duration = 10

        try:
            for i in range(num_iterations):
                print(f"Training iteration {i+1}/{num_iterations}")

                self.complete_task(task_description, num_iterations=10)

                self.train_local_model()

                generated_response = self.generate_local_response(prompt)
                print("Generated response:", generated_response)

                f1_score = self.evaluate_response(prompt, generated_response)
                print(f"F1 Score: {f1_score}")

                print(f"Sleeping for {sleep_duration} seconds...")
                time.sleep(sleep_duration)
        except Exception as e:
            print(f"Error during training and evaluation: {str(e)}")

# Example usage
GROQ_API_KEY = os.getenv("GROQ_API_KEY")                                                                                                                                                                                                               
api = OpenAIAPI(GROQ_API_KEY)
description = api.generate_response("Give me a prompt that a human may ask a chatbot to do. You are tasked with asking an AI to do a task for you based on what a user may want from the AI.", model="Gemma-7b-it")                                                                                                                           
task_description = description                                                                               
prompt = description                                                                

api.train_and_evaluate(task_description, prompt)
