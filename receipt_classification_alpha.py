import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import requests
import json
import torch
import torch.nn as nn
import numpy as np
import sys


def enhance_txt(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inv_gray = 255 - gray

    # Apply unsharp masking to the inverted grayscale image
    blurred = cv2.GaussianBlur(inv_gray, (5, 5), 0)
    unsharp_mask = cv2.addWeighted(inv_gray, 1.5, blurred, -0.5, 0)
    enhanced_inv = cv2.normalize(unsharp_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Apply the CLAHE algorithm to the inverted grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_inv_clahe = clahe.apply(enhanced_inv)

    # Invert the enhanced grayscale image
    enhanced = 255 - enhanced_inv_clahe

    return enhanced

# Load the image
image_path = sys.argv[1]
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to the grayscale image
#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (the receipt) and extract it
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
receipt = image[y:y+h, x:x+w]
'''
# Tilt the receipt to make it straight
edges = cv2.Canny(receipt, 100, 200)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
angle = 0
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle += np.arctan2(y2 - y1, x2 - x1)
angle /= len(lines)
rows, cols, _ = receipt.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), angle*180/np.pi, 1)
receipt = cv2.warpAffine(receipt, M, (cols, rows))
'''
# Enhance the text in the receipt
enhanced_receipt = enhance_txt(receipt)

# Use OCR to read the text from the image
text = pytesseract.image_to_string(enhanced_receipt, config="--oem 3 --psm 6 -l pol")

# Display the result using Matplotlib
print(text)
#plt.imshow(cv2.cvtColor(enhanced_receipt, cv2.COLOR_BGR2RGB))
#plt.show()

import re
lines = text.split("\n")

# Filter out empty lines and remove special characters from the lines (excluding specific characters)
filtered_lines = [re.sub(r"[^\w\s.,*-]", "", line) for line in lines if line.strip()]

# Join the filtered lines back into a single string
cleaned_text = "\n".join(filtered_lines)

print(cleaned_text)

def extract_products_from_receipt(receipt):
    lines = str(receipt).split("\n")  # Split the receipt into lines
    products = []
    prev_line = None
    prev_prev_line = None

    # Define the regex pattern for matching the desired lines
    pattern = r"(Szt|\bx[\d.]+|\*[.0-9]+|\d+(?:[,.\s]\d+)?(?:\s+\d+(?:[,.\s]\d+)?)+)"

    # Iterate through each line of the receipt
    for line in lines:
        if re.search(pattern, line, re.IGNORECASE):
            if len(line) < 30 and prev_line is not None:
                 products.append(prev_line + " " + line)
            else:
                 products.append(line)
        prev_line = line

    return products

products = extract_products_from_receipt(cleaned_text)
print(products)

def read_api_key_from_file(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

def get_text_embedding(text):17901790
    endpoint = "https://api.openai.com/v1/embeddings"
    model_id = "text-embedding-ada-002"
    api_key = read_api_key_from_file("api_key.txt")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "input": text,
        "model": model_id
    }

    response = requests.post(endpoint, headers=headers, data=json.dumps(data))
    response_data = response.json()

    if 'data' in response_data and len(response_data['data']) > 0:
        embedding = response_data['data'][0]['embedding']
        return embedding

    return None

# Creating a list of embeddings for each product
embeddings = []
for product in products:
    embedding = get_text_embedding(product)
    print(product)
    if embedding is not None:
        embeddings.append(embedding)

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Load the saved model's state dictionary
model = nn.Sequential(
    nn.Linear(in_features=1536, out_features=1536),
    nn.ReLU(),
    nn.Linear(in_features=1536, out_features=1536),
    nn.ReLU(),
    nn.Linear(in_features=1536, out_features=14)
).to(device)
model.load_state_dict(torch.load('C:/Users/tompys/Documents/Projekt/Paragony/model2.pth'))

# Set the model to evaluation mode
model.eval()

# Define the mapping of category numbers to labels
category_labels = {
    0: "Inne i przedmioty niesklasyfikowane",
    1: "Żywność",
    2: "Napoje",
    3: "Owoce i warzywa",
    4: "Mięso i ryby",
    5: "Nabiał",
    6: "Produkty zbożowe",
    7: "Słodycze i przekąski",
    8: "Napoje Alkoholowe",
    9: "Kosmetyki",
    10: "Produkty czystości",
    11: "Odzież",
    12: "Dom i ogród",
    13: "Medykamenty"
}

# Iterate over embeddings and products
output_list = []
with torch.no_grad():
    for embedding, product in zip(embeddings, products):
        # Convert the embedding to a tensor
        embedding_tensor = torch.tensor(embedding).unsqueeze(0).to(device)

        # Pass the embedding through the model
        output = model(embedding_tensor)
        
        # Apply softmax activation to obtain probabilities
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(output)
        
        # Get the predicted category
        predicted_category = torch.argmax(probabilities, dim=1).item()
        
        # Get the label corresponding to the predicted category
        label = category_labels.get(predicted_category, "Unknown")
        
        # Combine the label with the corresponding product
        result = f'{product}: {label}'

        # Append the result to the output list
        output_list.append(result)

# Convert the output list to a JSON string
json_output = json.dumps(output_list)

# Print or save the JSON output as desired
print(json_output)