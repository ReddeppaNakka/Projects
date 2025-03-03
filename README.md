Automated Endometriosis Detection Using CNN-RNN
This project automates the diagnosis of endometriosis using histopathological images through a hybrid CNN-RNN model. It combines the spatial feature extraction capabilities of Convolutional Neural Networks (CNNs) and sequential pattern recognition of Long Short-Term Memory (LSTM) networks. A Flask-based web application is provided for real-time image classification.

Features
Classifies histopathological images into four tissue types: EA (Endometrioid Adenocarcinoma), EH (Endometrial Hyperplasia), EP (Endometrial Polyp), and NE (Normal Endometrium).
Provides an easy-to-use web interface for uploading images and viewing predictions.
Displays predictions with confidence scores.
Steps to Use as follows
1. Clone the Repository
Clone the repository to your local machine:



git clone https://github.com/JinkaChenchuDharani/Automated-Endometriosis-Detection-Using-Histopathological-Image-Data-with-a-Hybrid-CNNRNN-Model/tree/main 


2. Install Dependencies
Install the required Python packages using pip:

pip install -r requirements.txt  
3. Save the Trained Model
Ensure the trained model (model.h5) is saved in the project directory. You can train and save the model using your dataset in Jupyter Notebook or any Python environment.

4. Run the Flask Application
Start the web application by running the following command in your terminal:

python app.py  
5. Access the Application
After running the application, a URL (e.g., http://127.0.0.1:5000/) will be displayed in the terminal. Open this URL in your browser.

6. Upload and Predict
Upload a histopathological image via the web interface.
The application will process the image and predict its endometriosis tissue type.

Prerequisites
Python 3.8 or higher
Flask
TensorFlow
Other dependencies listed in requirements.txt
How It Works
The saved model processes the uploaded histopathological image.
The image is passed through the hybrid CNN-RNN architecture to extract features and classify the tissue type.
The results, along with confidence scores, are displayed on the web interface.
Future Enhancements
Increase dataset size for better generalization.
Optimize the model for faster, real-time predictions.
Incorporate advanced techniques like attention mechanisms to improve accuracy.
