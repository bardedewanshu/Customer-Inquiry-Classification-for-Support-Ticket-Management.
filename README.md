# Customer-Inquiry-Classification-for-Support-Ticket-Management.
Project Overview
The Customer Inquiry Classification System is designed to automatically classify support tickets into predefined categories using Natural Language Processing (NLP) techniques. This helps in managing inquiries efficiently and providing timely responses to improve customer satisfaction.
Setup Instructions
Prerequisites
* Python 3.6+
* Internet connection for installing dependencies
Creating and Activating a Virtual Environment
1. Create Virtual Environment:
bash
python -m venv env
2. Activate Virtual Environment:
o On Windows:
bash
.\env\Scripts\activate
o On macOS/Linux:
bash
source env/bin/activate
Installing Required Libraries
bash
pip install pandas scikit-learn flask joblib matplotlib
Dataset
Overview
* File: support_tickets.csv
* Columns:
o text: The text of the customer inquiry.
o category: The category label for the inquiry.
Example Data
text
category
How can I reset my password?
Technical Support
When will my order arrive?
Order Status
I have a billing issue.
Billing
Data Preprocessing and Feature Extraction
Data Cleaning
* Convert text to lowercase.
* Remove punctuation and digits.
* Strip leading/trailing whitespace.
Feature Extraction
* Use TfidfVectorizer to convert text into numerical features.
* Example code:
python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
Model Training and Evaluation
Model
* Algorithm: Multinomial Naive Bayes
Training and Evaluation
* Split dataset into training and test sets.
* Train the model and evaluate using accuracy, precision, recall, and F1 score.
* Example code:
python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
ETL Pipeline
Extract
* Load data from support_tickets.csv.
Transform
* Clean and preprocess the text data.
* Apply feature extraction (TF-IDF).
Load
* Store processed data for analysis.
* Save the trained model and vectorizer.
Example Code
python
import pandas as pd

# Load dataset
df = pd.read_csv('support_tickets.csv')

# Data cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
Flask API
Setting Up Flask API
1. Install flask-ngrok:
bash
pip install flask-ngrok
2. Flask API Code:
python
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = clean_text(data['text'])
    vectorized_text = vectorizer.transform([text]).toarray()
    prediction = model.predict(vectorized_text)
    return jsonify({'category': prediction[0]})

if __name__ == '__main__':
    app.run()
Testing the API
* Use curl or Postman to send a POST request:
bash
curl -X POST http://<ngrok-url>/predict -H "Content-Type: application/json" -d '{"text": "your customer inquiry text here"}'
Visualization
Category Distribution
* Use matplotlib to visualize the distribution of support ticket categories.
* Example Code:
python
import matplotlib.pyplot as plt

df['category'].value_counts().plot(kind='bar')
plt.title('Category Distribution of Support Tickets')
plt.xlabel('Category')
plt.ylabel('Number of Tickets')
plt.show()
Usage
1. Run the Main Script:
bash
python customer_inquiry_classification.py
2. Interact with the API:
o Send POST requests to classify inquiries in real-time.
Acknowledgements
* Libraries: pandas, scikit-learn, flask, joblib, matplotlib.
* Tools: Google Colab for prototyping, VS Code for local development.

