import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file, redirect, url_for, Response
from PIL import Image
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder=".")

# Function to process the uploaded image and save it to a temporary file
def process_image(file):
    # Load the uploaded image
    image = Image.open(file)
    
    # Convert to OpenCV format (BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inv_gray_img = cv2.bitwise_not(gray_img)

    # Apply Gaussian blur to the inverted grayscale image
    blur_img = cv2.GaussianBlur(inv_gray_img, (39, 39), 0)

    # Invert the blurred image
    inv_blur_img = cv2.bitwise_not(blur_img)

    # Create a pencil sketch by dividing the grayscale image by the inverted blurred image
    pencil_img = cv2.divide(gray_img, inv_blur_img, scale=256.0)

    # Darken the edges (shading effect)
    shading_img = cv2.addWeighted(pencil_img, 1.5, cv2.GaussianBlur(pencil_img, (0, 0), 30), -0.5, 0)

    # Temporary folder for saving the processed image
    temp_folder = tempfile.mkdtemp()
    temp_filename = os.path.join(temp_folder, "processed_image.jpg")

    # Save the processed image to the temporary file
    cv2.imwrite(temp_filename, shading_img)

    return temp_filename  # Return the temporary file name

# Function to display the image using Matplotlib
def display_image(image_path):
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    return redirect(url_for('thank_you'))

@app.route('/')
def upload_form():
    # Render the HTML form for image upload
    with open('index.html', 'r') as f:
        html_content = f.read()
    return html_content


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    if file:
        try:
            # Process the uploaded image
            temp_image = process_image(file)

            # Display the processed image using Matplotlib
            display_image(temp_image)

            # Redirect to the "thank_you" route and pass the temporary image path
            return redirect(url_for('thank_you'))
        except Exception as e:
            return str(e)

@app.route('/thank_you')
def thank_you():
    
    with open('thank_you.html', 'r') as f:
        html_content = f.read()
    return html_content

if __name__ == '__main__':
    app.run(debug=True)
