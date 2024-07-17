
from openai import OpenAI
import logging
import time


# Setup logging to capture essential information and errors.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenAIHandler:

    def __init__(self, model='gpt-4o'):
        # Assuming the OPENAI_API_KEY is set in your environment variables for security.
        self.client = OpenAI()
        self.model = model

   # Creating Structured Messages
    def create_message(self, system_content, user_content, assistant_content=None):
        structured_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        if assistant_content:
            structured_messages.append({"role": "assistant", "content": assistant_content})
        return structured_messages

    # Getting a Response from OpenAI
    def get_response(self, messages):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            response_content = completion.choices[0].message.content
            logging.info(f"OpenAI Response Received:\n{response_content}")
            # Evaluate the necessity of this delay in your application context.
            time.sleep(10)
            return response_content
        except Exception as e:
            logging.error(f"Failed to get response from OpenAI: {e}")
            return None

# Additional Notes on usage
if __name__ == "__main__":
    # Initialize OpenAIHandler with a specific model
    handler = OpenAIHandler(model='gpt-4o')  
    system_message = "You are a python expert and an AI software engineer specializing in ML."
    user_message = "send me a cool LLM structure for training an LLM in python."

    messages = handler.create_message(system_content=system_message, user_content=user_message)
    response = handler.get_response(messages=messages)
    print(f"AI Response: {response}")
