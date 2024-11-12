# -*- coding: utf-8 -*-
"""Customer Inquiry .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HAScjQnL7MiLlqH0VpdtkUaOAcaRsqdf
"""

# Install required libraries
!pip install pandas scikit-learn flask joblib matplotlib

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/content/support_tickets.csv')

# Data cleaning function
def clean_text(title):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.strip()  # Remove leading/trailing whitespace
    return text

# Apply data cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Text tokenization and TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/content/support_tickets.csv')

# Data cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.strip()  # Remove leading/trailing whitespace
    return text

# Combine 'title' and 'body' into a single 'text' column
df['text'] = df['title'] + ' ' + df['body']

# Apply data cleaning to the combined 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Text tokenization and TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/content/support_tickets.csv')

# Data cleaning function
def clean_text(text):
    # Check if text is a string before applying string methods
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = text.strip()  # Remove leading/trailing whitespace
        return text
    else:
        # Handle non-string values (e.g., return empty string or NaN)
        return ''  # Or return float('nan') to preserve NaN values

# Combine 'title' and 'body' into a single 'text' column
df['text'] = df['title'] + ' ' + df['body']

# Apply data cleaning to the combined 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Text tokenization and TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

!pip install flask-ngrok

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

# Create Flask app
app = Flask(__name__)
run_with_ngrok(app)  # Starts ngrok when the app is run

# Load the model and vectorizer
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

from flask import Flask, request, jsonify
import joblib

# Load the trained model and vectorizer
model = joblib.load('support_ticket_model.pkl')
vectorizer = joblib.load('support_ticket_vectorizer.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Clean the input text
    cleaned_text = clean_text(text)

    # Vectorize the text
    text_vector = vectorizer.transform([cleaned_text]).toarray()

    # Predict the category
    prediction = model.predict(text_vector)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Change host to 0.0.0.0

from flask import Flask, request, jsonify
import joblib

# Load the trained model and vectorizer
# Use the correct filenames from when you saved them
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Assuming clean_text function is defined elsewhere in your code
    cleaned_text = clean_text(text)

    # Vectorize the text
    text_vector = vectorizer.transform([cleaned_text]).toarray()

    # Predict the category
    prediction = model.predict(text_vector)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Change host to 0.0.0.0

# Visualize category distribution
df['category'].value_counts().plot(kind='bar')
plt.title('Category Distribution of Support Tickets')
plt.xlabel('Category')
plt.ylabel('Number of Tickets')
plt.show()

curl -X POST http://<http://172.28.0.12:5000>/predict -H "Content-Type: application/json" -d '{"text": "your customer inquiry text here"}'