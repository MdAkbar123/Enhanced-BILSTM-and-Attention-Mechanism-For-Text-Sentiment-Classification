from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import os
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
app.secret_key = "sentimentanalysisapp"

# Config
MODEL_PATH = 'models/sentiment_model.keras'
TOKENIZER_PATH = 'models/tokenizer.pickle'
MAX_SEQUENCE_LENGTH = 150
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and tokenizer
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print("Error loading model/tokenizer:", e)
    traceback.print_exc()
    model = None
    tokenizer = None

users = {"admin": "password123", "user": "user123"}

@app.route('/')
def index():
    return redirect(url_for('project_info')) if 'username' in session else redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('project_info'))
    if request.method == 'POST':
        username, password = request.form['username'], request.form['password']
        if users.get(username) == password:
            session['username'] = username
            return redirect(url_for('project_info'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/project_info')
def project_info():
    return render_template('index.html', active_tab='home') if 'username' in session else redirect(url_for('login'))

def preprocess_text(text):
    if not text or pd.isna(text.strip()):
        raise ValueError("Text input is empty.")
    return pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

def predict_sentiment(text):
    if model is None or tokenizer is None:
        return "Model/tokenizer not loaded"
    try:
        pred = model.predict(preprocess_text(text))
        return ['negative', 'neutral', 'positive'][np.argmax(pred[0])]
    except Exception as e:
        print("Prediction error:", e)
        return "Prediction error"

@app.route('/single_analysis', methods=['GET', 'POST'])
def single_analysis():
    if 'username' not in session:
        return redirect(url_for('index'))
    text, result, error = '', None, None
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if not text:
            error = "Input text cannot be empty."
        else:
            result = predict_sentiment(text)
    return render_template('index.html', text=text, result=result, error=error, active_tab="single")

@app.route('/batch_analysis', methods=['GET', 'POST'])
def batch_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    results = None
    chart_data = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('batch_analysis'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return render_template('index.html', results=None, chart_data=None, error="No file selected.", active_tab="batch")
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                    df = pd.read_excel(filepath)
                else:
                    flash('Unsupported file format', 'error')
                    return render_template('index.html', results=None, chart_data=None, error="Unsupported file format.", active_tab="batch")
                
                # Check if 'Sentiment' column exists in the original data
                if 'Sentiment' not in df.columns:
                    flash("Original 'Sentiment' column not found in the dataset", 'warning')
                    df['Sentiment'] = 'N/A'  # Add placeholder if column doesn't exist
                
                predictions = []
                for _, row in df.iterrows():
                    try:
                        text = str(row.get("Text", ""))  # Ensure text is string
                        if text.strip():  # Only predict if text is not empty
                            pred = predict_sentiment(text)
                            # Convert prediction to match your original sentiment format if needed
                            pred = pred.capitalize()  # or other formatting
                        else:
                            pred = 'N/A'
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        pred = 'Error'
                    predictions.append(pred)
                
                # Add predictions to dataframe
                df['Sentiment_Predicted'] = predictions
                
                # Save the results with both original and predicted sentiment
                results_filename = f"results_{filename}"
                results_path = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
                df.to_csv(results_path, index=False)
                
                # Prepare data for display and chart
                results = df[['Text', 'Sentiment', 'Sentiment_Predicted']].to_dict('records')
                
                # Chart data - count predicted sentiments
                sentiment_counts = df['Sentiment_Predicted'].value_counts().to_dict()
                chart_data = {
                    'labels': list(sentiment_counts.keys()),
                    'values': list(sentiment_counts.values())
                }
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return render_template('index.html', results=None, chart_data=None, error=f"Error processing file: {str(e)}", active_tab="batch")
    
    return render_template('index.html', results=results, chart_data=chart_data, active_tab="batch")
@app.route('/download_results')
def download_results():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Find the latest results file
    results_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith('results_')]
    if not results_files:
        flash('No results available for download', 'error')
        return redirect(url_for('batch_analysis'))
    
    # Get the most recent file
    latest_file = max(results_files, key=lambda f: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], f)))
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_file)
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(debug=True)
