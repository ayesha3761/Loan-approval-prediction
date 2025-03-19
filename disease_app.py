import streamlit as st
import base64
st.title("Disease Prediction App")
st.write("Welcome to the Disease Prediction App!")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def add_bg_from_local(image_file):
    with open(image_file, 'rb') as f:
        image_data = f.read()
    b64_image = base64.b64encode(image_data).decode('utf-8')
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Provide the full path to the image on your Desktop
image_file = "medical_background.jpg"

add_bg_from_local(image_file)



# Title and Description
st.title('🩺 Disease Prediction App')
st.markdown('### Predict diabetes using the PIMA Indians Diabetes Dataset')
st.markdown('---')

# Load the dataset
@st.cache_data

def load_data():
q    data = pd.read_csv("diabetes.csv")
    return data

data = load_data()

# Display the dataset
if st.checkbox('Show Dataset'):
    st.write(data.head())

# Split the data into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_choice = st.selectbox('Select Model', ('Naive Bayes', 'Neural Network'))

if model_choice == 'Naive Bayes':
    model = GaussianNB()
elif model_choice == 'Neural Network':
    model = MLPClassifier(hidden_layer_sizes=(8,8), max_iter=500, random_state=42)

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display evaluation metrics
st.write(f'### Accuracy: {accuracy_score(y_test, y_pred):.2f}')
st.write('### Confusion Matrix')
st.write(confusion_matrix(y_test, y_pred))
st.write('### Classification Report')
st.text(classification_report(y_test, y_pred))

# Prediction form
st.sidebar.header('🔍 Predict a new case')
st.sidebar.markdown('Enter the patient details:')

pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=99, value=20)
insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=80)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=25.0)
dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=30)

if st.sidebar.button('Predict'):
    new_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(new_data)
    result = '🟢 Non-Diabetic' if prediction[0] == 0 else '🔴 Diabetic'
    st.sidebar.write(f'### Prediction: {result}')
