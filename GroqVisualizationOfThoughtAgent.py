import asyncio
import itertools
import os
from datetime import datetime
from colorama import init, Fore, Style
from termcolor import colored
import threading
import sys
import time
import subprocess
from groq import Groq

init()

def display_loading_animation(status_text, stop_event):
    animation = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
    while not stop_event.is_set():
        sys.stdout.write(Fore.YELLOW + '\r' + next(animation) + ' ' + status_text + Style.RESET_ALL)
        sys.stdout.flush()
        time.sleep(0.2)
    sys.stdout.write('\r' + ' ' * (len(status_text) + 2) + '\r')
    sys.stdout.flush()

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

async def generate_thought(question, messages):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    thought_response = client.chat.completions.create(
        messages=messages + [{'role': 'system', 'content': "Generate a thought related to the user's input, utilizing your 'Mind's Eye' for spatial reasoning."}],
        model="llama3-70b-8192",
    )
    return thought_response.choices[0].message.content

async def generate_visualization(thought, messages, prompt):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    visualization_response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'Visualize this thought: {thought} ONLY RESPOND WITH VISUALIZATION'}
        ],
        model="llama3-70b-8192",
    )
    return visualization_response.choices[0].message.content

async def refine_thought_with_visualization(thought, visualization, messages):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    refined_thought_response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': 'You are an AI that refines thoughts based on visualizations.'},
            {'role': 'user', 'content': f'Based on this visualization:\n{visualization}\nRefine the original thought: {thought}'}
        ],
        model="llama3-8b-8192",
    )
    return refined_thought_response.choices[0].message.content

async def generate_final_response(refined_thought, visualization1, visualization2, question, messages):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    final_response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': 'You are an AI assistant that generates responses based on a refined thought and two visualizations.'},
            {'role': 'user', 'content': f'Based on the refined thought: {refined_thought}, the first visualization:\n{visualization1}\nAnd the second visualization:\n{visualization2}\nGenerate a final response to the original question: {question}'}
        ],           
        model="llama3-8b-8192",
    )
    return final_response.choices[0].message.content

async def vot_chat(question, messages, end_word='quit'):
    if end_word == question.lower():
        return {'role': 'assistant', 'content': f'{Fore.YELLOW}Conversation ended.{Style.RESET_ALL} '}, True
    messages.append({'role': 'user', 'content': question})
    
    stop_event = threading.Event()
    loading_thread = threading.Thread(target=display_loading_animation, args=("Generating response...", stop_event))
    loading_thread.start()
    
    try:
        # Generate thought
        thought = await generate_thought(question, messages)
        print(f"{Fore.MAGENTA}Thought:{Style.RESET_ALL}")
        print(thought)
        print()
        
        # Generate first visualization
        visualization1 = await generate_visualization(thought, messages, 'You are an AI that generates visual representations of thoughts. Take the thought from your "Mind\'s Eye" and visualize it.')
        print(f"{Fore.GREEN}Visualization 1:{Style.RESET_ALL}")
        print(visualization1)
        print()
        
        # Refine thought with visualization
        refined_thought = await refine_thought_with_visualization(thought, visualization1, messages)
        print(f"{Fore.MAGENTA}Refined Thought:{Style.RESET_ALL}")
        print(refined_thought)
        print()
        
        # Generate second visualization
        visualization2 = await generate_visualization(refined_thought, messages, 'You are an AI that generates visual representations of thoughts. Take the refined thought and create a second visualization.')
        print(f"{Fore.GREEN}Visualization 2:{Style.RESET_ALL}")
        print(visualization2)
        print()
        
        # Generate final response
        full_response = await generate_final_response(refined_thought, visualization1, visualization2, question, messages)
              
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

async def main():
    messages = [{'role': 'system', 'content': 'You are a helpful AI assistant. Respond to the user\'s questions and assist them with their requests.'}]
    history_file = 'conversation_history.txt'
    await load_conversation_history(history_file, messages)
    
    print(f"{Fore.BLUE}Welcome to the Visualization-of-Thought AI assistant! (Type 'quit' to end the conversation){Style.RESET_ALL}")
    while True:
        user_input = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}")
        
        response, conversation_ended = await vot_chat(user_input, messages)
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
            response, conversation_ended = await vot_chat(new_prompt, messages)
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
        
        if user_input.lower() == 'clear history':
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
            print(f"{Fore.CYAN}- clear history: Clear the conversation history{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- save conversation: Save the current conversation to a file{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- help: Show this help message{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- quit: End the conversation{Style.RESET_ALL}")

asyncio.run(main())

