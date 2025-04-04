# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Load the trained model
# model = tf.keras.models.load_model("model/CNN_Trained_model.h5")

# # Define class labels
# class_labels = {
#     0: "Abnormal Heartbeat",
#     1: "History of MI",
#     2: "Myocardial Infarction",
#     3: "Normal Person"
# }

# # Function to preprocess the image
# def preprocess_image(img):
#     img = img.resize((227, 227))  # Resize to match model input size
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0  # Normalize pixel values
#     return img_array

# # Function to make predictions
# def predict_image(img):
#     img_array = preprocess_image(img)
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions, axis=1)[0]
#     confidence = np.max(predictions) * 100
#     return predicted_class, confidence, predictions

# # Streamlit App
# st.title("ECG Image Classification")
# st.write("Upload an ECG image to classify it into one of the following categories:")
# st.write(class_labels)

# # Upload image
# uploaded_file = st.file_uploader("Drag and drop an ECG image here", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     try:
#         # Open the uploaded image
#         img = Image.open(uploaded_file)

#         # Display the uploaded image
#         st.image(img, caption="Uploaded Image", use_column_width=True)

#         # Make prediction
#         predicted_class, confidence, predictions = predict_image(img)

#         # Display prediction result
#         st.subheader("Prediction Result")
#         st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
#         st.write(f"**Confidence:** {confidence:.2f}%")

#         # Plot the prediction probabilities
#         st.subheader("Prediction Probabilities")
#         fig, ax = plt.subplots()
#         ax.bar(class_labels.values(), predictions[0])
#         ax.set_xlabel("Class")
#         ax.set_ylabel("Probability")
#         ax.set_title("Prediction Probabilities for Each Class")
        
#         # Rotate x-axis labels vertically
#         plt.xticks(rotation=90)
        
#         st.pyplot(fig)

#     except Exception as e:
#         st.error(f"An error occurred: {e}")




# # Add the training and validation accuracy graph
# st.subheader("Training and Validation Accuracy")
# st.write("Below is the graph showing the training and validation accuracy over epochs:")
# st.image("Images/Accuracy graph.jpeg", use_column_width=True)

# # Model architecture and training details
# st.write("### Model Architecture")
# st.write("It uses LeakyReLU, Batch Normalization, Dropout, and Softmax for classification.")

# st.write("### Training Details")
# st.write("The model was trained with data augmentation, early stopping, and learning rate scheduling to achieve high accuracy and prevent overfitting.")

# # Add a footer
# st.markdown("---")
# st.write("Created by 🚀 Siddharth, Tanishiq, and Aditya")

import streamlit as st
import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import io
import dotenv
import os
from email.utils import formataddr
# ==================== CONFIGURATION ====================
# Email configuration - Replace these with your actual email service details
# Load environment variables
dotenv.load_dotenv()


EMAIL_CONFIG = {
    "sender_name": "CardiacInsight Team",
    "sender_email": os.getenv("GMAIL_ADDRESS"),  # Load from .env file
    "password": os.getenv("GMAIL_APP_PASSWORD"),  # App password, not your main password
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    
}

# ECG Class Labels
CLASS_LABELS = {
    0: "Abnormal Heartbeat",
    1: "History of MI",
    2: "Myocardial Infarction",
    3: "Normal Person"
}

# ==================== AUTHENTICATION SYSTEM ====================
def init_db():
    conn = sqlite3.connect('auth.db', check_same_thread=False)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 email TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (user_id INTEGER,
                  session_token TEXT,
                  expires_at TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    conn.commit()
    return conn

def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(32).hex()
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return f"{salt}${key.hex()}"

def verify_password(stored_password, provided_password):
    salt, key = stored_password.split('$')
    new_key = hash_password(provided_password, salt)
    return new_key == stored_password

def register_user(email, password):
    try:
        c = conn.cursor()
        if '@' not in email or '.' not in email.split('@')[1]:
            return False, "Invalid email format"
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        
        c.execute("SELECT email FROM users WHERE email = ?", (email,))
        if c.fetchone():
            return False, "Email already registered"
        
        hashed_pw = hash_password(password)
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", 
                 (email, hashed_pw))
        conn.commit()
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def login_user(email, password):
    try:
        c = conn.cursor()
        c.execute("SELECT id, password FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        
        if not user:
            return False, "Invalid credentials"
        
        user_id, stored_password = user
        if not verify_password(stored_password, password):
            return False, "Invalid credentials"
        
        session_token = hashlib.sha256(os.urandom(64)).hexdigest()
        expires_at = datetime.now() + timedelta(hours=2)
        
        c.execute("INSERT INTO sessions (user_id, session_token, expires_at) VALUES (?, ?, ?)",
                 (user_id, session_token, expires_at))
        conn.commit()
        
        return True, session_token
    except Exception as e:
        return False, f"Login error: {str(e)}"

# ==================== ECG ANALYSIS SYSTEM ====================
def load_model():
    try:
        model = tf.keras.models.load_model("model/CNN_Trained_model.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

def preprocess_image(img):
    img = img.resize((227, 227))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def predict_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class, np.max(predictions) * 100, predictions

# ==================== EMAIL SYSTEM ====================
def send_results_email(receiver_email, ecg_image, diagnosis, confidence, prob_plot):
    try:
        # Add this validation check
        if not EMAIL_CONFIG["sender_email"] or not EMAIL_CONFIG["password"]:
            raise ValueError("Email credentials not configured properly")
            
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG["sender_email"] 
        msg['To'] = receiver_email
        msg['Subject'] = "Your ECG Analysis Results"
        
        # HTML Email Body
        html = f"""<html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2a52be;">ECG Analysis Report</h2>
                    <p>Dear User,</p>
                    
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                        <h3 style="margin-top: 0; color: #d9534f;">Diagnosis: {diagnosis}</h3>
                        <p>Confidence Level: <strong>{confidence:.2f}%</strong></p>
                    </div>
                    
                    <p>This automated analysis should be reviewed by a medical professional.</p>
                    
                    <h3>Your ECG Image:</h3>
                    <img src="cid:ecg_image" style="max-width: 100%; border: 1px solid #ddd;">
                    
                    <h3>Probability Distribution:</h3>
                    <img src="cid:prob_chart" style="max-width: 100%; border: 1px solid #ddd;">
                    
                    <div style="margin-top: 30px; font-size: 0.9em; color: #6c757d;">
                       
                        <p>This is an automated message - please do not reply directly.</p>
                    </div>
                </div>
            </body>
        </html>"""
        
        msg.attach(MIMEText(html, 'html'))
        
        # Attach ECG Image
        img_io = io.BytesIO()
        ecg_image.save(img_io, format='PNG')
        img_io.seek(0)
        img_attachment = MIMEImage(img_io.read())
        img_attachment.add_header('Content-ID', '<ecg_image>')
        msg.attach(img_attachment)
        
        # Attach Probability Chart
        chart_io = io.BytesIO()
        prob_plot.savefig(chart_io, format='png', bbox_inches='tight', dpi=300)
        chart_io.seek(0)
        chart_attachment = MIMEImage(chart_io.read())
        chart_attachment.add_header('Content-ID', '<prob_chart>')
        msg.attach(chart_attachment)
        
        # Send email with error handling
        try:
            with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
                server.ehlo()
                server.starttls()
                server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["password"])
                server.sendmail(
                    EMAIL_CONFIG["sender_email"],
                    receiver_email,
                    msg.as_string()
                )
            return True
        except smtplib.SMTPAuthenticationError:
            st.error("Authentication failed. Please check your email credentials.")
            return False
        except Exception as e:
            st.error(f"Email sending failed: {str(e)}")
            return False
            
    except Exception as e:
        st.error(f"Error preparing email: {str(e)}")
        return False

# ==================== STREAMLIT UI ====================
def auth_ui():
    st.title("ECG Analysis Portal")
    menu = st.radio("Select Action", ["Login", "Register"])
    
    if menu == "Login":
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            success, message = login_user(email, password)
            if success:
                st.session_state.update({
                    'authenticated': True,
                    'email': email,
                    'session_token': message
                })
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(message)
    
    elif menu == "Register":
        st.subheader("Create New Account")
        new_email = st.text_input("Email Address")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords don't match!")
            else:
                success, message = register_user(new_email, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

def main_app():
    st.title("ECG Image Classification")
    st.write("Upload an ECG image to classify it into one of the following categories:")
    st.write(CLASS_LABELS)

    uploaded_file = st.file_uploader("Drag and drop an ECG image here", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            predicted_class, confidence, predictions = predict_image(img)

            st.subheader("Prediction Result")
            st.write(f"**Predicted Class:** {CLASS_LABELS[predicted_class]}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            # Plot the prediction probabilities
            st.subheader("Prediction Probabilities")
            fig, ax = plt.subplots()
            ax.bar(CLASS_LABELS.values(), predictions[0])
            ax.set_xlabel("Class")
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Probabilities for Each Class")
            plt.xticks(rotation=90)
            st.pyplot(fig)

            # Email section with configuration check
            st.subheader("Email Results")
            
            if not all([EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["password"], EMAIL_CONFIG["smtp_server"]]):
                st.warning("Email service is not configured. Please contact the administrator.")
            else:
                if st.button("Send Results to My Email"):
                    with st.spinner("Sending your report..."):
                        if send_results_email(
                            st.session_state.email,
                            img,
                            CLASS_LABELS[predicted_class],
                            confidence,
                            fig
                        ):
                            st.success("Report sent successfully to your registered email!")
                        else:
                            st.error("Failed to send email. Please try again later.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.subheader("Training and Validation Accuracy")
    st.write("Below is the graph showing the training and validation accuracy over epochs:")
    st.image("Images/Accuracy graph.jpeg", use_column_width=True)

    st.write("### Model Architecture")
    st.write("It uses LeakyReLU, Batch Normalization, Dropout, and Softmax for classification.")

    st.write("### Training Details")
    st.write("The model was trained with data augmentation, early stopping, and learning rate scheduling to achieve high accuracy and prevent overfitting.")

    st.markdown("---")
    st.write("Created by 🚀 Siddharth, Tanishiq, and Aditya")
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.sidebar.write(f"Logged in as: {st.session_state.email}")

# ==================== MAIN APP FLOW ====================
if __name__ == "__main__":
    # Initialize database and model
    conn = init_db()
    model = load_model()

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        auth_ui()
    else:
        main_app()
