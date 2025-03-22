import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model/CNN_Trained_model.h5")

# Define class labels
class_labels = {
    0: "Abnormal Heartbeat",
    1: "History of MI",
    2: "Myocardial Infarction",
    3: "Normal Person"
}

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((227, 227))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence, predictions

# Streamlit App
st.title("ECG Image Classification")
st.write("Upload an ECG image to classify it into one of the following categories:")
st.write(class_labels)

# Upload image
uploaded_file = st.file_uploader("Drag and drop an ECG image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the uploaded image
        img = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        predicted_class, confidence, predictions = predict_image(img)

        # Display prediction result
        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Plot the prediction probabilities
        st.subheader("Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(class_labels.values(), predictions[0])
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities for Each Class")
        
        # Rotate x-axis labels vertically
        plt.xticks(rotation=90)
        
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")




# Add the training and validation accuracy graph
st.subheader("Training and Validation Accuracy")
st.write("Below is the graph showing the training and validation accuracy over epochs:")
st.image("Images/accuracy.png", use_column_width=True)

# Model architecture and training details
st.write("### Model Architecture")
st.write("The model used is a custom CNN with two branches (stack and full) and 38 layers in total.")
st.write("It uses LeakyReLU, Batch Normalization, Dropout, and Softmax for classification.")

st.write("### Training Details")
st.write("The model was trained with data augmentation, early stopping, and learning rate scheduling to achieve high accuracy and prevent overfitting.")

# Add a footer
st.markdown("---")
st.write("Created by 🚀 Siddharth, Tanishiq, and Aditya")


