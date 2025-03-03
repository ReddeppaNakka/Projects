# app.py

import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Define the path to the uploads folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the trained Keras model
try:
    model = load_model('optimized_cnn_rnn_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class labels mapped to folder names
class_labels = {
    'Class1': 'Folder1',
    'Class2': 'Folder2',
    'Class3': 'Folder3',
    'Class4': 'Folder4'
}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess the image to match the model's expected input."""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Error in preprocessing image: {e}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle image upload and prediction."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in the request.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected for uploading.")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
            except Exception as e:
                return render_template('index.html', error=f"Error saving file: {e}")
            
            try:
                processed_image = preprocess_image(file_path)
                prediction = model.predict(processed_image)
                predicted_class = list(class_labels.keys())[np.argmax(prediction)]
                predicted_folder = class_labels[predicted_class]
                confidence = np.max(prediction) * 100
            except Exception as e:
                return render_template('index.html', error=f"Error during prediction: {e}")
            
            return render_template('result.html',
                                   filename=filename,
                                   prediction=predicted_folder,
                                   confidence=round(confidence, 2))
    
    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    """Display the uploaded image."""
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if model:
        app.run(debug=True)
    else:
        print("Model could not be loaded. Please check the model file.")
