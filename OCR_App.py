# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from keras.models import load_model
import joblib
import pandas as pd

# Initialize Streamlit app
st.title("Gujarati Handwritten OCR App")
st.write("Upload an image containing Gujarati characters to detect and label them.")

# Load models and encoders
character_model = load_model("Character_model_gray_v2.h5")
character_label_decoder = joblib.load("Character_label_encoder_gray_v2.joblib")
consonant_model = load_model("Consonant_model_gray_v2.h5")
consonant_label_decoder = joblib.load("Consonant_label_encoder_gray_v2.joblib")
vowel_model = load_model("Vowel_model_gray_v2.h5")
vowel_label_decoder = joblib.load("Vowel_label_encoder_gray_v2.joblib")

# Load additional resources
gujarati_consonants_dict = {
    'k': 'ક', 'kh': 'ખ', 'g': 'ગ', 'gh': 'ઘ', 'ng': 'ઙ',
    'ch': 'ચ', 'chh': 'છ', 'j': 'જ', 'z': 'ઝ', 'at': 'ટ', 
    'ath': 'ઠ', 'ad': 'ડ', 'adh': 'ઢ', 'an': 'ણ', 't': 'ત', 
    'th': 'થ', 'd': 'દ', 'dh': 'ધ', 'n': 'ન', 'p': 'પ', 'f': 'ફ', 
    'b': 'બ', 'bh': 'ભ', 'm': 'મ', 'y': 'ય', 'r': 'ર', 'l': 'લ', 
    'v': 'વ', 'sh': 'શ', 'shh': 'ષ', 's': 'સ', 'h': 'હ', 
    'al': 'ળ', 'ks': 'ક્ષ', 'gn': 'જ્ઞ'
}

gujarati_vowels_dict = {'a': 'આ', 'i': 'ઇ', 'ee': 'ઈ', 'u': 'ઉ',
    'oo': 'ઊ', 'ri': 'ઋ', 'rii': 'ૠ', 'e': 'એ', 'ai': 'ઐ',
    'o': 'ઓ', 'au': 'ઔ', 'amn': 'અં', 'ah': 'અઃ', "ru": "અૃ", "ra": "અ્ર",
    'ar': "્રઅ"}

df = pd.read_csv("barakshari.csv", index_col=0)

# Function to preprocess the image
def image_preprocessing(img_array):
    processed_img = cv2.resize(img_array, (50, 50))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)
    processed_img = processed_img / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)
    return processed_img

# Function to get Gujarati label
def get_gujarati_label(class_label, gujarati_dict):
    guj_class_label = ""
    if class_label.lower() in gujarati_dict.keys():
        guj_class_label = gujarati_dict[class_label.lower()]
    return guj_class_label

# Prediction function
def predict_character(cropped_img):
    processed_img = image_preprocessing(cropped_img)
    consonant_prediction = consonant_model.predict(processed_img)
    consonant_predicted_class = np.argmax(consonant_prediction)
    consonant_label = consonant_label_decoder.inverse_transform([consonant_predicted_class])[0]
    consonant_guj_label = get_gujarati_label(consonant_label, gujarati_consonants_dict)

    vowel_prediction = vowel_model.predict(processed_img)
    vowel_predicted_class = np.argmax(vowel_prediction)
    vowel_label = vowel_label_decoder.inverse_transform([vowel_predicted_class])[0]
    vowel_guj_label = get_gujarati_label(vowel_label, gujarati_vowels_dict)

    character_prediction = character_model.predict(processed_img)
    character_predicted_class = np.argmax(character_prediction)
    character_label = character_label_decoder.inverse_transform([character_predicted_class])[0]
    character_guj_label = df.loc[consonant_guj_label.strip(), vowel_guj_label.strip()]

    return f'{consonant_guj_label} {vowel_guj_label} {character_guj_label}'

# Function to apply thresholding
def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# Streamlit file upload functionality
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 1: Thresholding the image
    thresh_img = thresholding(image)

    # Step 2: Dilating the image
    kernel = np.ones((5, 85), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)

    # Step 3: Finding contours
    (contours, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Step 4: Cropping and detecting characters
    words_list = []
    img_cropped = image.copy()

    for line in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(line)
        roi_line = image[y:y+h, x:x+w]
        (cnt, _) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

        for word in sorted_contour_words:
            if cv2.contourArea(word) < 400:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])
            cv2.rectangle(img_cropped, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 255, 100), 2)

    # Displaying cropped image with bounding boxes
    st.image(img_cropped, caption="Image with Bounding Boxes", use_column_width=True)

    # Load font for Gujarati characters
    font_path = "NotoSansGujarati-VariableFont_wdth,wght.ttf"
    font = ImageFont.truetype(font_path, 40)

    # Convert OpenCV image to PIL format for adding text
    img_pil = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))

    # Step 5: Predicting characters and labeling them on the image
    for word_coords in words_list:
        x1, y1, x2, y2 = word_coords
        cropped_character = image[y1:y2, x1:x2]

        # Get predicted label
        predicted_label = predict_character(cropped_character)

        # Initialize ImageDraw for drawing on the image
        draw = ImageDraw.Draw(img_pil)
        draw.text((x1, y1 - 30), predicted_label, font=font, fill=(255, 0, 0))

    # Convert back to OpenCV format to display final image
    img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Display final image with labeled characters
    st.image(img_with_text, caption="Image with Gujarati Character Labels", use_column_width=True)
