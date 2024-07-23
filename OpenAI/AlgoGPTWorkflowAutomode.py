import subprocess
import logging
import time
from openai import OpenAI
import ast
import re
import json

gpt4 = "gpt-4o"
gpt3 = "gpt-3.5-turbo"

class OpenAIHandler:
    def __init__(self, model=gpt4):
        self.client = OpenAI()
        self.model = model

    def create_message(self, system_content, user_content, assistant_content=None):
        structured_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        if assistant_content:
            structured_messages.append({"role": "assistant", "content": assistant_content})
        return structured_messages

    def get_response(self, messages):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            response_content = completion.choices[0].message.content
            logging.info("OpenAI Response Received:\n{}".format(response_content))
            time.sleep(10)
            return response_content
        except Exception as e:
            logging.error(f"Failed to get response from OpenAI: {e}")
            return None

class AlgoDeveloper:
    def __init__(self, openai_handler):
        self.openai_handler = openai_handler

    def develop_algo(self, algo_code=None, error_message=None):
        system_message, user_message = self._generate_messages(algo_code, error_message)
        messages = self.openai_handler.create_message(system_message, user_message)
        response = self.openai_handler.get_response(messages)
        improved_algo_code = CodingUtils.extract_python_code(response)
        if improved_algo_code and CodingUtils.is_code_valid(improved_algo_code):
            logging.info("AI algorithm improvement found and validated.")
            return improved_algo_code
        logging.error("No valid Python code improvements found in the response.")
        return algo_code

    def _generate_messages(self, algo_code, error_message):
        if not algo_code:
            system_message = "Develop a new AI algorithm based on organic learning principles, adhering to Python standards."
            user_message = "Create an initial AI that learns like a stem cell within the Python coding environment."
        else:
            system_message = "Improve the AI algorithm based on the following code and feedback, adhering to Python standards."
            user_message = f"The current AI algorithm is:\n{algo_code}\nError encountered: {error_message}\nHow should the algorithm be improved?"
        return system_message, user_message

class AlgoTester:
    def __init__(self, openai_handler):
        self.openai_handler = openai_handler

    def test_algo(self, algo_code):
        try:
            test_process = subprocess.Popen(["python", "-c", algo_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = test_process.communicate(timeout=30)
            if stderr:
                logging.error(f"Algorithm Testing Failed: {stderr}")
                return False, stderr
            logging.info(f"Algorithm Testing Success: {stdout}")
            return True, stdout
        except subprocess.TimeoutExpired:
            logging.error("Algorithm testing timed out.")
            return False, "Algorithm testing timed out."
        except Exception as e:
            logging.error(f"Error in testing algorithm: {e}")
            return False, str(e)

class CodingUtils:
    @staticmethod
    def is_code_valid(code):
        try:
            ast.parse(code)
            logging.info("Python code validation passed.")
            return True
        except SyntaxError as e:
            logging.error(f"Syntax error in the generated code: {e}")
            return False

    @staticmethod
    def extract_python_code(markdown_text):
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        python_code_blocks = [match.strip() for match in matches]
        return python_code_blocks[0] if python_code_blocks else ""

class FileManager:
    @staticmethod
    def save_script(filename, content):
        with open(filename, 'w') as file:
            file.write(content)
            logging.info(f"Algorithm script saved to {filename} successfully.")

    @staticmethod
    def save_conversation_dataset(filename, conversation_history):
        with open(filename, 'w') as file:
            json.dump(conversation_history, file, indent=4)
            logging.info(f"Conversation history saved to {filename} successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    openai_handler = OpenAIHandler(model=gpt4)  # You can switch to gpt3 if needed
    algo_developer = AlgoDeveloper(openai_handler)
    algo_tester = AlgoTester(openai_handler)
    
    algo_code = None
    max_iterations = 100
    error_message = None
    performance_metrics = {}
    conversation_history = []

    logging.info("Starting the iterative improvement process for the AI algorithm.")
    for iteration in range(max_iterations):
        logging.info(f"Iteration {iteration + 1}: Developing and testing the algorithm.")
        conversation_history.append({
            'iteration': iteration,
            'algorithm_code': algo_code,
            'error_message': error_message
        })
        algo_code = algo_developer.develop_algo(algo_code, error_message)
        if algo_code:
            test_result, feedback = algo_tester.test_algo(algo_code)
            if test_result:
                logging.info(f"Algorithm successfully tested. Feedback: {feedback}")
                performance_metrics[iteration] = feedback
                conversation_history[-1]['feedback'] = feedback
            else:
                logging.error(f"Algorithm testing failed. Error: {feedback}")
                error_message = feedback
                conversation_history[-1]['error'] = feedback
        else:
            logging.error("Failed to develop a valid algorithm. Stopping the iterative process.")
            break

    FileManager.save_script('final_algo_script.py', algo_code)
    FileManager.save_conversation_dataset('conversation_dataset.json', conversation_history)
    logging.info("Iterative improvement process completed.")
    logging.info(f"Final performance metrics: {performance_metrics}")
