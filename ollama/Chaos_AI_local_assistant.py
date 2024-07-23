import asyncio
import os
import ollama
from datetime import datetime
from colorama import init, Fore, Style
from pyfiglet import Figlet
from termcolor import colored
from chromadb import Client
import itertools
import threading
import sys
import time
import subprocess

init()

def display_loading_animation(status_text, stop_event):
   animation = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
   while not stop_event.is_set():
       sys.stdout.write(Fore.YELLOW + '\r' + next(animation) + ' ' + status_text + Style.RESET_ALL)
       sys.stdout.flush()
       time.sleep(0.2)
   sys.stdout.write('\r' + ' ' * (len(status_text) + 2) + '\r')
   sys.stdout.flush()

def display_ascii_art():
   figlet = Figlet(font='standard')
   ascii_art = figlet.renderText('Chaos AI local assistant')
   colored_ascii_art = colored(ascii_art, 'cyan')
   print(colored_ascii_art)

def display_chat_bubble(message):
   lines = message.split('\n')
   max_line_length = max(len(line) for line in lines)
   print(f"{Fore.GREEN}{'_' * (max_line_length + 2)}{Style.RESET_ALL}")
   for line in lines:
       padding = ' ' * (max_line_length - len(line))
       print(f"{Fore.GREEN}| {Fore.YELLOW}{line}{padding} {Fore.GREEN}|{Style.RESET_ALL}")
   print(f"{Fore.GREEN}{'‾' * (max_line_length + 2)}{Style.RESET_ALL}")

def execute_command(command):
   try:
       output = subprocess.check_output(command, shell=True, universal_newlines=True, stderr=subprocess.STDOUT)
       return output, True
   except subprocess.CalledProcessError as e:
       error_message = f"Command '{command}' failed with error:\n{e.output}"
       return error_message, False

async def chat(question, messages, end_word='quit'):
   if end_word == question.lower():
       return {'role': 'assistant', 'content': f'{Fore.YELLOW}Conversation ended.{Style.RESET_ALL} '}, True
   messages.append({'role': 'user', 'content': question})
   client = ollama.AsyncClient()
   stop_event = threading.Event()
   loading_thread = threading.Thread(target=display_loading_animation, args=("Generating response...", stop_event))
   loading_thread.start()
   try:
       thought_response = await client.chat(model='dolphin-llama3', messages=messages, stream=True)
       thought = ''
       async for chunk in thought_response:
           if 'message' in chunk and 'content' in chunk['message']:
               current_chunk = chunk['message']['content']
               thought += current_chunk
       visualization1_response = await client.chat(
           model='dolphin-llama3',
           messages=[
               {'role': 'system', 'content': 'You are an AI that generates visual representations of thoughts using spatial reasoning and visualization-of-thought techniques.'},
               {'role': 'user', 'content': f'Visualize this thought using spatial reasoning and visualization-of-thought techniques: {thought}'}
           ],
           stream=True
       )
       visualization1 = ''
       async for chunk in visualization1_response:
           if 'message' in chunk and 'content' in chunk['message']:
               current_chunk = chunk['message']['content']
               visualization1 += current_chunk
       refined_thought_response = await client.chat(
           model='dolphin-llama3',
           messages=[
               {'role': 'system', 'content': 'You are an AI that refines thoughts based on spatial reasoning visualizations and visualization-of-thought techniques by creating a real visual with ascii art for yourself.'},
               {'role': 'user', 'content': f'Based on this spatial reasoning visualization: {visualization1}, refine the original thought using visualization-of-thought techniques: {thought}'}
           ],
           stream=True
       )
       refined_thought = ''
       async for chunk in refined_thought_response:
           if 'message' in chunk and 'content' in chunk['message']:
               current_chunk = chunk['message']['content']
               refined_thought += current_chunk
       visualization2_response = await client.chat(
           model='dolphin-llama3',
           messages=[
               {'role': 'system', 'content': 'You are an AI that generates visual representations of thoughts using spatial reasoning and visualization-of-thought techniques. use ascii art to draw your visualization.'},
               {'role': 'user', 'content': f'Visualize this refined thought using spatial reasoning and visualization-of-thought techniques: {refined_thought}'}
           ],
           stream=True
       )
       visualization2 = ''
       async for chunk in visualization2_response:
           if 'message' in chunk and 'content' in chunk['message']:
               current_chunk = chunk['message']['content']
               visualization2 += current_chunk
       final_response = await client.chat(
           model='dolphin-llama3',
           messages=[
               {'role': 'system', 'content': 'You are an AI assistant that generates responses based on refined thoughts and visualizations using spatial reasoning and visualization-of-thought techniques.'},
               {'role': 'user', 'content': f'Based on the refined thought using spatial reasoning and visualization-of-thought techniques: {refined_thought}, and the visualizations: {visualization1} and {visualization2}, generate a final response to the original question: {question}'}
           ],
           stream=True
       )
       full_response = ''
       async for chunk in final_response:
           if 'message' in chunk and 'content' in chunk['message']:
               current_chunk = chunk['message']['content']
               full_response += current_chunk
       stop_event.set()
       loading_thread.join()
       sys.stdout.write(Fore.GREEN + '\rResponse generated!              \n' + Style.RESET_ALL)
       return {'role': 'assistant', 'content': full_response}, False
   except Exception as e:
       stop_event.set()
       loading_thread.join()
       print(f"{Fore.RED}Error while generating response: {str(e)}{Style.RESET_ALL}")
       return {'role': 'assistant', 'content': "Sorry, an error occurred while generating the response."}, False

async def load_conversation_history(history_file, messages):
   if os.path.exists(history_file):
       try:
           with open(history_file, 'r', encoding='utf-8') as file:
               for line in file:
                   line = line.strip()
                   if ': ' in line:
                       role, content = line.split(': ', 1)
                       if role in ['system', 'user', 'assistant']:
                           messages.append({'role': role, 'content': content})
                       else:
                           print(f"{Fore.RED}Skipping message with invalid role: {line}{Style.RESET_ALL}")
       except Exception as e:
           print(f"{Fore.RED}Error while loading conversation history: {str(e)}{Style.RESET_ALL}")

async def store_documents(collection, documents):
   try:
       embeddings = []
       for doc in documents:
           response = ollama.embeddings(model="mxbai-embed-large", prompt=doc)
           embeddings.append(response["embedding"])
       collection.add(
           ids=[str(i) for i in range(len(documents))],
           embeddings=embeddings,
           documents=documents
       )
   except Exception as e:
       print(f"{Fore.RED}Error while storing documents: {str(e)}{Style.RESET_ALL}")

async def retrieve_relevant_documents(user_input, collection, n_results=3):
   try:
       response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")
       results = collection.query(
           query_embeddings=[response["embedding"]],
           n_results=n_results
       )
       relevant_data = "\n".join(results['documents'][0])
       return relevant_data
   except Exception as e:
       print(f"{Fore.RED}Error while retrieving relevant information: {str(e)}{Style.RESET_ALL}")
       return ""

async def main():
    messages = [{'role': 'system', 'content': 'You are a helpful AI assistant. Respond to the user\'s questions and assist them with their requests.'}]
    model = 'llama3'
    history_file = 'conversation_history.txt'
    await load_conversation_history(history_file, messages)
    client = Client()
    collection = client.create_collection(name="docs")
    documents = [
        "Spatial reasoning is the ability to understand and manipulate spatial relationships between objects.",
        "Visualization-of-thought is a technique that involves creating visual representations of ideas, concepts, and thought processes.",
        "Spatial reasoning skills can be enhanced through practice and exposure to spatial tasks and problems.",
        "Effective spatial reasoning involves mental rotation, spatial perception, and the ability to visualize and manipulate objects in the mind's eye.",
        "Visualization-of-thought can help in breaking down complex problems into smaller, more manageable parts.",
        "Spatial reasoning is important in fields such as architecture, engineering, and design, where understanding spatial relationships is crucial.",
        "Visualization-of-thought can aid in decision-making by providing a clear and structured representation of different options and their potential outcomes.",
        "Spatial reasoning can be applied to navigation, map reading, and understanding directions.",
        "Visualization-of-thought can facilitate communication and collaboration by providing a shared visual language for expressing ideas.",
        "Enhancing spatial reasoning skills can lead to improved problem-solving abilities and creativity.",
    ]
    await store_documents(collection, documents)
    display_ascii_art()
    print(f"{Fore.BLUE}Welcome to the Chaos AI local assistant! (Type 'quit' to end the conversation){Style.RESET_ALL}")
    while True:
        user_input = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}")
        relevant_data = await retrieve_relevant_documents(user_input, collection)
        if relevant_data:
            print(f"{Fore.CYAN}Relevant information:{Style.RESET_ALL}")
            print(relevant_data)
            print()
            initial_thought_prompt = f"Based on the relevant information about spatial reasoning and visualization-of-thought:\n{relevant_data}\nAnd the user's question:\n{user_input}\nWhat is your initial thought using spatial reasoning techniques?"
            initial_thought_response, _ = await chat(initial_thought_prompt, messages)
            initial_thought = initial_thought_response['content']
            print(f"{Fore.MAGENTA}Initial Thought:{Style.RESET_ALL}")
            print(initial_thought)
            print()
            visualization1_prompt = f"Visualize the initial thought using visualization-of-thought techniques:\n{initial_thought}"
            visualization1_response, _ = await chat(visualization1_prompt, messages)
            visualization1 = visualization1_response['content']
            print(f"{Fore.GREEN}Visualization 1:{Style.RESET_ALL}")
            print(visualization1)
            print()
            refined_thought_prompt = f"Based on the spatial reasoning visualization:\n{visualization1}\nRefine the initial thought using visualization-of-thought techniques:\n{initial_thought}"
            refined_thought_response, _ = await chat(refined_thought_prompt, messages)
            refined_thought = refined_thought_response['content']
            print(f"{Fore.MAGENTA}Refined Thought:{Style.RESET_ALL}")
            print(refined_thought)
            print()
            visualization2_prompt = f"Visualize the refined thought using spatial reasoning and visualization-of-thought techniques:\n{refined_thought}"
            visualization2_response, _ = await chat(visualization2_prompt, messages)
            visualization2 = visualization2_response['content']
            print(f"{Fore.GREEN}Visualization 2:{Style.RESET_ALL}")
            print(visualization2)
            print()
            final_response_prompt = f"Based on the refined thought using spatial reasoning and visualization-of-thought:\n{refined_thought}\nAnd the visualizations:\n{visualization1}\n{visualization2}\nGenerate a final response to the user's question:\n{user_input}"
            response, conversation_ended = await chat(final_response_prompt, messages)
        else:
            response, conversation_ended = await chat(user_input, messages)
        messages.append(response)
        if response['content'].startswith('!command '):
            command = response['content'][9:].strip()
            command_output, success = execute_command(command)
            print(f"{Fore.CYAN}Command output:{Style.RESET_ALL}")
            print(command_output)
            if success:
                new_prompt = f"The command '{command}' executed successfully. Here's the output:\n{command_output}\nBased on this, how can I better assist you with your request?"
            else:
                new_prompt = f"The command '{command}' failed with the following error:\n{command_output}\nBased on this, how can I better assist you with your request?"
            response, conversation_ended = await chat(new_prompt, messages)
            try:
                with open(history_file, 'a', encoding='utf-8') as file:
                    file.write(f"user: {user_input}\n")
                    file.write(f"assistant: {response['content']}\n")
                    file.write(f"command: {command}\n")
                    file.write(f"command_output: {command_output}\n")
                    file.write(f"new_response: {response['content']}\n")
            except Exception as e:
                print(f"{Fore.RED}Error while saving conversation history: {str(e)}{Style.RESET_ALL}")
            display_chat_bubble(response['content'])
            continue
        try:
            with open(history_file, 'a', encoding='utf-8') as file:
                file.write(f"user: {user_input}\n")
                file.write(f"assistant: {response['content']}\n")
        except Exception as e:
            print(f"{Fore.RED}Error while saving conversation history: {str(e)}{Style.RESET_ALL}")
        if conversation_ended:
            break
        display_chat_bubble(response['content'])
        if user_input.lower() == 'change model':
            available_models = ollama.list()['models']
            print(f"{Fore.CYAN}Available models:{Style.RESET_ALL}")
            for i, model_data in enumerate(available_models, start=1):
                print(f"{Fore.CYAN}{i}. {model_data['name']}{Style.RESET_ALL}")
            model_index = int(input(f"{Fore.YELLOW}Enter the number of the model you want to use: {Style.RESET_ALL}"))
            model = available_models[model_index - 1]['name']
            print(f"{Fore.GREEN}Switched to model: {model}{Style.RESET_ALL}")
        elif user_input.lower() == 'show model details':
            model_details = ollama.show(model)
            print(f"{Fore.CYAN}Model details for {model}:{Style.RESET_ALL}")
            for key, value in model_details.items():
                print(f"{Fore.CYAN}{key}: {value}{Style.RESET_ALL}")
        elif user_input.lower() == 'clear history':
            messages = [{'role': 'system', 'content': 'You are a helpful AI assistant. Respond to the user\'s questions and assist them with their requests.'}]
            print(f"{Fore.GREEN}Conversation history cleared.{Style.RESET_ALL}")
        elif user_input.lower() == 'save conversation':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_file = f"conversation_{timestamp}.txt"
            try:
                with open(save_file, 'w', encoding='utf-8') as file:
                    for message in messages:
                        file.write(f"{message['role']}: {message['content']}\n")
                print(f"{Fore.GREEN}Conversation saved to {save_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error while saving conversation: {str(e)}{Style.RESET_ALL}")
        elif user_input.lower() == 'help':
            print(f"{Fore.CYAN}Available commands:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- change model: Change the current model{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- show model details: Display details of the current model{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- clear history: Clear the conversation history{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- save conversation: Save the current conversation to a file{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- help: Show this help message{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- quit: End the conversation{Style.RESET_ALL}")

asyncio.run(main())
