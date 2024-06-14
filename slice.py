import cv2
import openai
import base64
import numpy as np
import time
import os
from PIL import ImageGrab
import pytesseract
import asyncio
import concurrent.futures

# Function to capture the screen and return as a BGR image
def capture_screen():
    screenshot = ImageGrab.grab()
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    return screenshot_bgr

# Function to detect edges and find contours
def detect_edges_and_find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def save_image(image, file_path):
    """
    Saves an image to a file.

    Parameters:
    - image: np.ndarray, the image data as a NumPy array.
    - file_path: str, the path to save the image file.

    Returns:
    None
    """
    cv2.imwrite(file_path, image)

def analyse(image_path, prompt, api_key):
    openai.api_key = api_key

    # Load the image and convert it to a base64 encoded string
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the input with both the prompt and the image
    combined_prompt = f"{prompt}\n\n[image: {encoded_image}]"

    try:

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": combined_prompt}
            ]
        )

        # Extract the response content from the first message
        val = response['choices'][0]['message']['content']
        return val

    except Exception as e:
        print(f"Could not get a response for: {image_path}. Error: {e}")
        return None


def is_character(x):
    """Check if the variable x is a single character."""
    return isinstance(x, str) and len(x) == 1

def is_string(x):
    """Check if the variable x is a non-empty string."""
    return isinstance(x, str) and len(x) > 0

def is_text(x):
    """Check if the variable x is either a single character or a non-empty string."""
    return is_character(x.strip()) or is_string(x.strip())

# Function to check if a region contains text with a timeout
def contains_text(roi, count):
    try:
        text = pytesseract.image_to_string(roi, timeout=3)  # Set timeout for 3 seconds
        if(is_text(text)):
            print("(" + text.strip() + ")")
            return True
        return bool(text.strip())
    except pytesseract.pytesseract.TesseractError:
        return False

# Function to draw bounding boxes around contours excluding those containing text
async def draw_bounding_boxes(image, contours):
    trueimage = image
    count = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Define a square region around the contour
            square_size = max(w, h)
            x_center = x + w // 2
            y_center = y + h // 2
            x_square = max(0, x_center - square_size // 2)
            y_square = max(0, y_center - square_size // 2)

            MAX = 20
            if(w > MAX and h > MAX):
                # Ensure the square size is valid for resizing
                if square_size > 0 and x_square + square_size <= image.shape[1] and y_square + square_size <= image.shape[0]:
                    roi = image[y_square:y_square + square_size, x_square:x_square + square_size]
                    # Downscale the ROI to speed up OCR processing
                    try:
                        val = executor.submit(contains_text, roi, count)
                        futures.append(val)
                        valid_contours.append((x_square, y_square, square_size))
                    except:
                        print("Error")
                        pass
        for future, (x_square, y_square, square_size) in zip(futures, valid_contours):
            print(count)
            count = count + 1
            # if not future.result():
            # Draw the bounding box if no text is found
            roi = image[y_square:y_square + square_size, x_square:x_square + square_size]
            cv2.rectangle(image, (x_square, y_square), (x_square + square_size, y_square + square_size), (0, 255, 0), 2)
            # Example usage
            api_key = "your-key"
            image_path = "captured_image-" + str(time.time()) + ".png"  # Specify the path to save the image
            try:
                if( not contains_text(roi,count)):
                    save_image(roi, image_path)
                    #hint = os.name()
                    prompt = "Lets Play a Game: Computer Vision, Mission Critical. You are going to be given image snippets from a desktop. Goal: select a name for each snippet. Make sure your response is a single word. If you're unsure just respond with 'Not Available'."

                    val = analyse(image_path, prompt, api_key)
                    try:
                        position = ((x_square+square_size+10), y_square+20)  # Coordinates (x, y) of the text starting point

                        # Choose the font type and scale
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5

                        # Choose the font and text color
                        font_color = (0, 255, 0)  # BGR format (Blue, Green, Red)

                        # Draw the text on the image
                        cv2.putText(image, val, position, font, font_scale, font_color, thickness=2)

                        #if val != 'Not Available':
                        print("Response: " + str(val) + " " + image_path)
                    except Exception as ee:
                        print(f"Loop exception. EE-> Error: {ee}")
                        pass
            except Exception as e:
                print(f"Loop exception. Error: {e}")
                pass

# Main function to capture, detect, and draw bounding boxes
def main():
    # Capture the screen
    screen_image = capture_screen()

    # Detect edges and find contours
    contours = detect_edges_and_find_contours(screen_image)

    # Draw bounding boxes around the contours excluding those containing text
    asyncio.run(draw_bounding_boxes(screen_image, contours))

    # Display the image with bounding boxes
    cv2.imshow('Icon and Button Detection', screen_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
