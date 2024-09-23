import cv2
import numpy as np
from PIL import Image
# import pytesseract

# Function to load the image
def read_image(image_path):
    image = cv2.imread(image_path)

    # Set maximum width and height constraints
    max_width = 1200
    max_height = 800

    # Get the current width and height of the image
    h, w = image.shape[:2]

    # Calculate the resizing ratio while maintaining aspect ratio
    if w > h:
        ratio = max_width / w
    else:
        ratio = max_height / h

    # Calculate the new dimensions
    new_width = int(w * ratio)
    new_height = int(h * ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

# Function to detect bounding boxes around characters
def detect_characters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use thresholding or adaptive thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to detect characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store bounding boxes of each character
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small boxes (noise) by size
        if w > 10 and h > 10:  # You can adjust the size
            bounding_boxes.append((x, y, w, h))
    
    return bounding_boxes

# Function to crop characters
def crop_characters(image, bounding_boxes):
    cropped_images = []
    for box in bounding_boxes:
        x, y, w, h = box
        cropped_images.append(image[y:y+h, x:x+w])
    return cropped_images

# Function to send to model for prediction
def predict_character(model, cropped_image):
    # Preprocess the image as per the model’s requirements
    # Assuming the model returns the predicted label
    # Example: prediction = model.predict(cropped_image)
    # For now, let's mock the prediction
    return "M"

# Function to process the image, predict and label the characters
def process_image(image_path, model):
    image = read_image(image_path)
    bounding_boxes = detect_characters(image)
    cropped_images = crop_characters(image, bounding_boxes)

    for i, box in enumerate(bounding_boxes):
        label = predict_character(model, cropped_images[i])

        # Draw rectangle and label around the character
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save or display the labeled image
    cv2.imwrite("labeled_image.jpg", image)
    cv2.imshow("Labeled Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Assuming you have a trained model loaded here
model = None  # Replace with your model

# Run the OCR process
image_path = "Character_samples.jpg"
process_image(image_path, model)
