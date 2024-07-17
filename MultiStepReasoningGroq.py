import os
import json
import time
import subprocess
import requests
from groq import Groq
from typing import List, Dict, Union, Callable
from functools import wraps

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = 'llama3-70b-8192'

def validate_arguments(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if not isinstance(arg, (str, int, float, bool, list, dict)):
                raise ValueError(f"Invalid argument type: {type(arg)}")
        for value in kwargs.values():
            if not isinstance(value, (str, int, float, bool, list, dict)):
                raise ValueError(f"Invalid argument type: {type(value)}")
        return func(*args, **kwargs)
    return wrapper

@validate_arguments
def base_function_call_method(function_name: str, args: Dict[str, Union[str, int, float, bool, List, Dict]]) -> Union[str, int, float, bool, List, Dict]:
    if function_name == 'fibonacci':
        return fibonacci(args['n'])
    elif function_name == 'run_command':
        return run_command(args['command'])
    elif function_name == 'write_to_file':
        return write_to_file(args['filename'], args['content'])
    elif function_name == 'read_from_file':
        return read_from_file(args['filename'])
    elif function_name == 'make_api_request':
        return make_api_request(args['url'], args['method'], args.get('params'), args.get('data'), args.get('headers'))


    else:
        raise ValueError(f"Unknown function: {function_name}")

@validate_arguments
def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

@validate_arguments
def run_command(command: str) -> str:
    print(f"Running command: {command}")
    try:
        output = subprocess.check_output(command, shell=True, universal_newlines=True, stderr=subprocess.STDOUT)
        print("Command executed successfully.")
        print(f"Command output: {output}")
        return output
    except subprocess.CalledProcessError as e:
        error_message = f"Error: {e.output}"
        print(error_message)
        return error_message
@validate_arguments
def write_to_file(filename: str, content: str) -> str:
    print(f"Attempting to write to file: {filename}")
    try:
        with open(filename, 'w') as file:
            print(f"File {filename} opened successfully.")
            file.write(content)
            print(f"Content written to file {filename}.")
        return f"Successfully wrote to file: {filename}"
    except IOError as e:
        error_message = f"Error writing to file {filename}: {str(e)}"
        print(error_message)
        return error_message

@validate_arguments
def read_from_file(filename: str) -> str:
    try:
        with open(filename, 'r') as file:
            content = file.read()
        return content
    except IOError as e:
        return f"Error reading file: {str(e)}"

@validate_arguments
def make_api_request(url: str, method: str, params: Dict[str, str] = None, data: Dict[str, str] = None, headers: Dict[str, str] = None) -> str:
    try:
        response = requests.request(method, url, params=params, data=data, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error making API request: {str(e)}"








@validate_arguments
def reasoning_loop(user_prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    try:
        # 1. Comprehension
        messages.insert(0, {"role": "system", "content": "Analyze the problem statement and identify the key aspects of the task."})
        comprehension_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        messages.append({"role": "assistant", "content": comprehension_response.choices[0].message.content})

        # 2. Planning
        messages.insert(0, {"role": "system", "content": "Develop a plan to solve the problem, considering different approaches and algorithms."})
        planning_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        messages.append({"role": "assistant", "content": planning_response.choices[0].message.content})

        # 3. Coding
        messages.insert(0, {"role": "system", "content": "Implement the planned solution in code, following best practices and conventions."})
        coding_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        code = coding_response.choices[0].message.content
        messages.append({"role": "assistant", "content": code})

        # 4. Testing
        messages.insert(0, {"role": "system", "content": "Develop test cases to verify the correctness of the implemented code."})
        testing_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        test_cases = testing_response.choices[0].message.content
        messages.append({"role": "assistant", "content": test_cases})

        # 5. Debugging and Refinement
        messages.insert(0, {"role": "system", "content": "Debug the code, identify any issues, and refine the implementation."})
        debugging_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        refined_code = debugging_response.choices[0].message.content
        messages.append({"role": "assistant", "content": refined_code})

        # 6. Review and Documentation
        messages.insert(0, {"role": "system", "content": "Review the final code, add comments, and provide documentation."})
        review_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        documented_code = review_response.choices[0].message.content
        messages.append({"role": "assistant", "content": documented_code})

        # 7. Analysis
        messages.insert(0, {"role": "system", "content": "Analyze the response and determine if it satisfies the requirements. If not, provide feedback on which step needs improvement."})
        analysis_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        analysis_result = analysis_response.choices[0].message.content
        messages.append({"role": "assistant", "content": analysis_result})

        while "satisfactory" not in analysis_result.lower():
            # Determine the step that needs improvement based on the analysis result
            if "comprehension" in analysis_result.lower():
                step = 1
            elif "planning" in analysis_result.lower():
                step = 2
            elif "coding" in analysis_result.lower():
                step = 3
            elif "testing" in analysis_result.lower():
                step = 4
            elif "debugging" in analysis_result.lower():
                step = 5
            elif "review" in analysis_result.lower() or "documentation" in analysis_result.lower():
                step = 6
            else:
                step = 0

            if step != 0:
                # Send the response back to the respective step for improvement
                messages.insert(0, {"role": "system", "content": f"The response needs improvement. Please revisit step {step} and provide an updated response."})
                update_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=4096
                )
                messages.append({"role": "assistant", "content": update_response.choices[0].message.content})

                # Analyze the updated response
                messages.insert(0, {"role": "system", "content": "Analyze the updated response and determine if it satisfies the requirements. If not, provide feedback on which step needs further improvement."})
                analysis_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=4096
                )
                analysis_result = analysis_response.choices[0].message.content
                messages.append({"role": "assistant", "content": analysis_result})
            else:
                break

        # 8. Feedback Loop
        messages.insert(0, {"role": "system", "content": "Reflect on the problem-solving process and provide insights for improvement."})
        feedback_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        messages.append({"role": "assistant", "content": feedback_response.choices[0].message.content})

        return messages[-1]["content"]

    except Exception as e:
        print(f"Error in reasoning_loop: {str(e)}")
        return "An error occurred while processing your request."

if __name__ == "__main__":
    user_prompt = "Create a fun RPG game with Python that includes character creation, combat system, and inventory management."
    print("User prompt:", user_prompt)
    time.sleep(1)
    result = reasoning_loop(user_prompt)
    print("Assistant response:", result)
