import random
import string

def generate_credentials():
    username = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    password = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=15))
    return username, password

def automate_task():

    
    username, password = generate_credentials()
    
    print(f"Username: {username}")
    print(f"Password: {password}")
    
    
    print("Task automation completed successfully.")

if __name__ == "__main__":
    automate_task()
