from flask import Flask, render_template, url_for, request
import sqlite3

from PIL import Image
import numpy as np
import cv2

import numpy as np

import shutil
import os
import sys
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from skimage import io, transform
from joblib import load

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
# Load the trained model
clf = load('breast_cancer_prediction_rf_model.joblib')

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

def analyse(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    test_image = io.imread(path)
    test_image_resized = transform.resize(test_image, (50, 50)).flatten()

    # Predict the output class and probability
    output_class = clf.predict([test_image_resized])[0]
    probabilities = clf.predict_proba([test_image_resized])[0]

    # Print the output class and probability
    print(f'Output Class: {"MALIGNANT" if output_class == 1 else "BENIGN"}')
    print(f'Probability: {probabilities[output_class]:.2f}')
    acc=f"{probabilities[output_class]:.2f}"
    return output_class,acc

def calculate_shapes(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Convert the image to grayscale if it's a color image
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=-1)

    # Threshold the image to get a binary image
    threshold = 128  # Adjust the threshold as needed
    binary_image = (image_array > threshold).astype(np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the number of shapes (contours)
    num_shapes = len(contours)

    return num_shapes    

def calculate_largest_blur_area(image_path, img):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for light pink color
    lower_pink = np.array([140, 50, 180])  # Adjust these values based on your specific shade of pink
    upper_pink = np.array([170, 255, 255])

    # Create a mask for the pink color
    pink_mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    # Find contours in the mask
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the largest contour (area)
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    # Blur the largest area in the original image
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
        image[y:y+h, x:x+w] = blurred_roi

        # Save the result
        cv2.imwrite('static/blur/'+img, image)

    return largest_area

def calculate_largest_area_in_mm(image_path, pixel_to_mm_conversion):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for light pink color
    lower_pink = np.array([140, 50, 180])  # Adjust these values based on your specific shade of pink
    upper_pink = np.array([170, 255, 255])

    # Create a mask for the pink color
    pink_mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    # Find contours in the mask
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the largest contour (area)
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    # Calculate the largest area in square millimeters
    largest_area_mm = largest_area * pixel_to_mm_conversion**2

    return largest_area_mm

def count_abnormalities_in_largest_area(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for light pink color
    lower_pink = np.array([140, 50, 180])  # Adjust these values based on your specific shade of pink
    upper_pink = np.array([170, 255, 255])

    # Create a mask for the pink color
    pink_mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    # Find contours in the mask
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the largest contour (area)
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    # Count the number of abnormalities in the largest area
    num_abnormalities = 0
    if largest_contour is not None:
        for contour in contours:
            # Check if the contour is not the largest contour and intersects with it
            if contour is not largest_contour:
                # Calculate the intersection area
                intersection_area = cv2.contourArea(cv2.convexHull(contour))
                if intersection_area > 0:
                    num_abnormalities += 1

    return num_abnormalities
    
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        print("DEBUG: POST request received to /detect")
        print(f"DEBUG: Request files: {list(request.files.keys())}")
        
        # Handle file upload
        if 'img' not in request.files:
            print("DEBUG: No 'img' file in request")
            return render_template('userlog.html', error="No file selected")
        
        file = request.files['img']
        print(f"DEBUG: File object: {file}")
        print(f"DEBUG: File filename: {file.filename}")
        
        if file.filename == '':
            print("DEBUG: Empty filename")
            return render_template('userlog.html', error="No file selected")
        
        if file:
            # Save the uploaded file
            filename = str(file.filename)
            path = os.path.join('static/test', filename)
            file.save(path)
            
            images = []
            results = []
            class_names=['Benign', 'Malignant']
            cls_id,ac = analyse(path)
            acc=float(ac)*100
            print(f"\n\n Class id : {class_names[cls_id]}  \n\n accuracy : {acc}")
            print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[cls_id], acc))
            images.append(f"http://127.0.0.1:5000/static/test/{filename}")
            results.append("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[cls_id],  acc))

            num_shapes = calculate_shapes(path)
            print(f"Number of shapes: {num_shapes}")
            results.append(f"Number of shapes: {num_shapes}")

            largest_blur_area = calculate_largest_blur_area(path, filename)
            print(f"Largest blur area: {largest_blur_area}")
            images.append(f"http://127.0.0.1:5000/static/blur/{filename}")
            results.append(f"Largest blur area: {largest_blur_area}")

            pixel_to_mm_conversion = 0.1  # Replace with your pixel-to-mm conversion factor
            largest_area_mm = calculate_largest_area_in_mm(path, pixel_to_mm_conversion)
            print(f"Largest area in square millimeters: {largest_area_mm} mm")
            results.append(f"Largest area in square millimeters: {largest_area_mm} mm")

            num_abnormalities = count_abnormalities_in_largest_area(path)
            print(f"Number of lymph nodes in the metastasis area: {num_abnormalities}")
            results.append(f"Number of lymph nodes in the metastasis area: {num_abnormalities}")

            image1 = cv2.imread(path)
            gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"static/gray/{filename}", gray_image)
            images.append(f"http://127.0.0.1:5000/static/gray/{filename}")

            #apply the Canny edge detection
            edges = cv2.Canny(image1, 100, 200)
            cv2.imwrite(f"static/edges/{filename}", edges)
            images.append(f"http://127.0.0.1:5000/static/edges/{filename}")

            #apply thresholding to segment the image
            retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
            cv2.imwrite(f"static/threshold/{filename}", threshold2)
            images.append(f"http://127.0.0.1:5000/static/threshold/{filename}")

            return render_template('userlog.html', images=images, results=results)

    return render_template('userlog.html')

@app.route('/detect2', methods=['GET', 'POST'])
def detect2():
    if request.method == 'POST':
        # Handle file upload
        if 'img2' not in request.files:
            return render_template('userlog.html', error="No file selected")
        
        file = request.files['img2']
        if file.filename == '':
            return render_template('userlog.html', error="No file selected")
        
        if file:
            # Save the uploaded file
            filename = str(file.filename)
            path = os.path.join('static/test', filename)
            file.save(path)
            
            images = []
            results = []
            class_names=['BENIGN', 'MALIGNANT']
            cls_id,ac = analyse(path)
            acc=float(ac)*100
            print(f"\n\n Class id : {class_names[cls_id]}  \n\n accuracy : {acc}")
            print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[cls_id], acc))
            images.append("http://127.0.0.1:5000/static/test/"+filename)
            results.append("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[cls_id],  acc))

            num_shapes = calculate_shapes(path)
            print(f"Number of shapes: {num_shapes}")
            results.append(f"Number of shapes: {num_shapes}")

            largest_blur_area = calculate_largest_blur_area(path, filename)
            print(f"Largest blur area: {largest_blur_area}")
            images.append("http://127.0.0.1:5000/static/blur/"+filename)
            results.append(f"Largest blur area: {largest_blur_area}")

            pixel_to_mm_conversion = 0.1  # Replace with your pixel-to-mm conversion factor
            largest_area_mm = calculate_largest_area_in_mm(path, pixel_to_mm_conversion)
            print(f"Largest area in square millimeters: {largest_area_mm} mm")
            results.append(f"Largest area in square millimeters: {largest_area_mm} mm")

            num_abnormalities = count_abnormalities_in_largest_area(path)
            print(f"Number of lymph nodes in the metastasis area: {num_abnormalities}")
            results.append(f"Number of lymph nodes in the metastasis area: {num_abnormalities}")

            image1 = cv2.imread(path)
            gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('static/gray/'+filename, gray_image)
            images.append("http://127.0.0.1:5000/static/gray/"+filename)

            #apply the Canny edge detection
            edges = cv2.Canny(image1, 100, 200)
            cv2.imwrite('static/edges/'+filename, edges)
            images.append("http://127.0.0.1:5000/static/edges/"+filename)

            #apply thresholding to segment the image
            retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
            cv2.imwrite('static/threshold/'+filename, threshold2)
            images.append("http://127.0.0.1:5000/static/threshold/"+filename)

            return render_template('userlog.html', images1=images, results1=results)

    return render_template('userlog.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
