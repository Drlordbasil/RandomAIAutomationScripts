import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont

# Define the ASCII characters to use, ordered based on brightness
ASCII_CHARS = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]

def resize_image(image, new_width=100, adjust_aspect_ratio=1.0):
    """
    Resize the image while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = height / float(width)
    new_height = int(new_width * aspect_ratio * adjust_aspect_ratio)
    resized_image = image.resize((new_width, new_height))
    return resized_image

def grayscale_image(image):
    """
    Convert the image to grayscale.
    """
    return image.convert("L")

def pixels_to_ascii(image, line_width=100):
    """
    Convert pixel values to ASCII characters.
    """
    pixels = image.getdata()
    ascii_chars = []
    for i, pixel_value in enumerate(pixels):
        ascii_index = pixel_value // 25
        ascii_index = max(0, min(ascii_index, len(ASCII_CHARS) - 1))  # Clamp the index within valid range
        ascii_chars.append(ASCII_CHARS[ascii_index])
        if (i + 1) % line_width == 0:
            ascii_chars.append('\n')  # Add a newline character after every line_width characters
    return "".join(ascii_chars)

def save_ascii_art(ascii_art, file_format):
    """
    Save the ASCII art as an image or text file.
    """
    file_path = filedialog.asksaveasfilename(title="Save ASCII Art", filetypes=[(file_format, f"*.{file_format}")])
    if file_path:
        try:
            if file_format == "txt":
                with open(file_path, "w") as file:
                    file.write(ascii_art)
                print(f"ASCII art saved as {file_path}")
            else:
                # Create an image with the ASCII art
                lines = ascii_art.split("\n")
                font = ImageFont.truetype("arial.ttf", 12)
                max_width = max(len(line) for line in lines)
                height = len(lines) * font.getsize("")[1]
                image = Image.new("RGB", (max_width * font.getsize("")[0], height), color=(255, 255, 255))
                draw = ImageDraw.Draw(image)
                y = 0
                for line in lines:
                    draw.text((0, y), line, font=font, fill=(0, 0, 0))
                    y += font.getsize("")[1]
                image.save(file_path)
                print(f"ASCII art saved as {file_path}")
        except Exception as e:
            print(f"Error saving ASCII art: {e}")

def open_image():
    """
    Open a file dialog to select the input image.
    """
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.png *.gif")])
    if file_path:
        try:
            image = Image.open(file_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            return

        # Preprocess the image
        resized_image = resize_image(image, new_width=300, adjust_aspect_ratio=1.5)
        grayscale_image_result = grayscale_image(resized_image)
        ascii_art = pixels_to_ascii(grayscale_image_result, line_width=resized_image.width)
        save_ascii_art(ascii_art, file_format="txt")
        save_ascii_art(ascii_art, file_format="png")
        # Display the ASCII art
        text_box.delete('1.0', tk.END)  # Clear the text box
        text_box.insert(tk.END, ascii_art)



# Create the main window
root = tk.Tk()
root.title("ASCII Art Generator")
root.geometry("800x600")  # Set a larger initial window size

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(side=tk.TOP, padx=10, pady=10)



# Create a text box to display the ASCII art
text_box = tk.Text(root, font=("Courier", 3), wrap=tk.NONE, width=300, height=400)
text_box.pack(side=tk.TOP, padx=10, pady=10)



# Start the main event loop
root.mainloop()
