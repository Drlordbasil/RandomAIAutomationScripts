import logging
import time
from openai import OpenAI

class RevisedStoryCollaborator:
    def __init__(self, model="gpt-4o", story_file="story.txt"):
        self.client = OpenAI()
        self.model = model
        self.story_file = story_file
        # Initialize with a narrator description
        self.characters = ["Narrator"]  # List of characters, starting with Narrator
        self.story_context = []  # To hold the evolving story context

    def add_character(self, character_name, description):
        """Introduce a new character with a description provided by the narrator."""
        if character_name not in self.characters:
            self.characters.append(character_name)  # Add new character
            # Narrator introduces the character
            introduction = f"[Narrator] {character_name} enters the story: {description}"
            self.story_context.append(introduction)
            self.save_to_story(introduction)

    def save_to_story(self, content):
        """Append the latest contribution to the story file."""
        with open(self.story_file, "a") as file:
            file.write(f"{content}\n")

    def generate_narration(self, scene_description):
        """Narrator sets the scene."""
        narration = f"[Narrator] {scene_description}"
        self.story_context.append(narration)
        self.save_to_story(narration)

    def character_interaction(self, character_name, prompt):
        """Generate character-specific interaction based on the current story context."""
        # Ensure the character is part of the story
        if character_name in self.characters:
            # Create the prompt by combining the story context with the specific prompt
            full_prompt = "\n".join(self.story_context[-5:]) + f"\n[{character_name}] {prompt}"
            messages = [{"role": "system", "content": full_prompt}]
            
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=150  # Adjust based on needs
                )
                response = completion.choices[0].message.content.strip()
                interaction = f"[{character_name}] {response}"
                self.story_context.append(interaction)
                self.save_to_story(interaction)
            except Exception as e:
                logging.error(f"Failed to generate interaction for {character_name}: {e}")
        else:
            logging.error(f"{character_name} is not a recognized character in the story.")

    def run_story(self):
        """Main loop to dynamically develop the story with character and narrator contributions."""
        # Example: Narrator sets the initial scene
        self.generate_narration("In a land far away, an adventure begins.")
        
        # Example interactions - replace with dynamic input or structured story flow
        self.character_interaction("Alice", "What do we do now?")
        self.character_interaction("Bob", "Let's explore the castle ruins to the east.")
class StoryFormatter:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def format_story(self):
        """Reads the dialogue-based story and converts it into a chapter book format."""
        with open(self.input_file, 'r') as infile, open(self.output_file, 'w') as outfile:
            for line in infile:
                # Process narrator lines
                if line.startswith("[Narrator]"):
                    narration = line.replace("[Narrator]", "").strip().capitalize()
                    outfile.write(f"{narration}\n\n")

                # Process dialogue lines
                elif "[" in line and "]" in line:
                    parts = line.strip().split("] ")
                    if len(parts) > 1:
                        characters_part, dialogue = parts[0], parts[1]
                        characters = characters_part.replace("[", "")
                        # Correct handling for multiple characters
                        if characters:
                            formatted_characters = " and ".join(characters.split(", ")).strip()
                            formatted_dialogue = f'{formatted_characters} said, "{dialogue.capitalize()}"\n\n'
                            outfile.write(formatted_dialogue)
                        else:
                            # If characters are mentioned but no dialogue follows
                            outfile.write(f'{characters} seems to ponder silently.\n\n')
                    else:
                        # Handle any malformed lines by writing them directly to maintain story integrity
                        outfile.write(f"{line.strip().capitalize()}\n\n")
                else:
                    # Directly write any lines that don't match expected patterns
                    outfile.write(f"{line.strip().capitalize()}\n\n")

    def display_formatted_story(self):
        """Displays the formatted story from the output file."""
        with open(self.output_file, 'r') as outfile:
            print(outfile.read())

# Example setup and usage
if __name__ == "__main__":
    story_collaborator = RevisedStoryCollaborator()
    story_collaborator.add_character("Alice", "A curious explorer with a keen sense of adventure.")
    story_collaborator.add_character("Bob", "A wise mage knowledgeable in the arcane arts.")

    # Begin the story development
    story_collaborator.run_story()
    formatter = StoryFormatter('story.txt', 'formatted_story.txt')
    formatter.format_story()
    formatter.display_formatted_story()
