import os
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# --- 1. INITIALIZATION ---
app = Flask(__name__)
app.secret_key = 'secret_clinical_key'

# Database Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'heart_disease.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- 2. LOAD AI MODELS ---
try:
    rf_model = joblib.load('heart_rf_model.pkl')
    svm_model = joblib.load('heart_svm_model.pkl')
    nn_model = joblib.load('heart_nn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

# --- 3. DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class HeartPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    age = db.Column(db.Float)
    sex = db.Column(db.Float)
    cp = db.Column(db.Float)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    fbs = db.Column(db.Float)
    restecg = db.Column(db.Float)
    thalach = db.Column(db.Float)
    exang = db.Column(db.Float)
    oldpeak = db.Column(db.Float)
    slope = db.Column(db.Float)
    ca = db.Column(db.Float)
    thal = db.Column(db.Float)
    prediction_result = db.Column(db.String(50))
    probability = db.Column(db.Float)

# --- 4. CHATBOT CONFIGURATION ---
# The exact order your model was trained on
QUESTIONS = [
    ("age", "What is your age?"),
    ("sex", "What is your sex? (1 for Male, 0 for Female)"),
    ("cp", "Chest Pain type? (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)"),
    ("trestbps", "Resting Blood Pressure (mm Hg)?"),
    ("chol", "Serum Cholesterol (mg/dl)?"),
    ("fbs", "Fasting Blood Sugar > 120 mg/dl? (1: True, 0: False)"),
    ("restecg", "Resting ECG results? (0: Normal, 1: Abnormality, 2: Hypertrophy)"),
    ("thalach", "Maximum Heart Rate achieved?"),
    ("exang", "Exercise Induced Angina? (1: Yes, 0: No)"),
    ("oldpeak", "ST depression (e.g., 1.5)?"),
    ("slope", "Slope of peak exercise ST segment? (0: Up, 1: Flat, 2: Down)"),
    ("ca", "Number of major vessels (0-3)?"),
    ("thal", "Thalassemia? (1: Normal, 2: Fixed, 3: Reversible)")
]

# --- 5. ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form.get('username')
        uemail = request.form.get('email')
        upswd = request.form.get('password')
        if not uemail: return "Email is required!"
        hashed_pw = generate_password_hash(upswd)
        new_user = User(username=uname, email=uemail, password=hashed_pw)
        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
        except Exception as e:
            return f"Error: {e}"
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form.get('username')
        upass = request.form.get('password')
        user = User.query.filter_by(username=uname).first()
        if user and check_password_hash(user.password, upass):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        return "Invalid credentials."
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST']) # Added both methods here
def predict():
    if request.method == 'GET':
        # This part loads the chatbot page when you first visit /predict
        return render_template('predict.html')
    
    # This part handles the "go" button from your JavaScript
    try:
        # 1. Get data from the form
        feature_list = [
            float(request.form.get('age')),
            float(request.form.get('sex')),
            float(request.form.get('cp')),
            float(request.form.get('trestbps')),
            float(request.form.get('chol')),
            float(request.form.get('fbs')),
            float(request.form.get('restecg')),
            float(request.form.get('thalach')),
            float(request.form.get('exang')),
            float(request.form.get('oldpeak')),
            float(request.form.get('slope')),
            float(request.form.get('ca')),
            float(request.form.get('thal'))
        ]
        print(f"DEBUG: Raw Input List -> {feature_list}")
        # 2. Scale and Predict (Using nn_model as in your earlier code)
        final_features = scaler.transform([feature_list])
        print(f"DEBUG: Scaled Data -> {final_features}")
        prediction = nn_model.predict(final_features)[0]
        probability_array = nn_model.predict_proba(final_features)
        print(f"DEBUG: Raw Prob Array -> {probability_array}")
        probability = probability_array[0][1]
        
        # 3. Save to Session for results.html
        session['last_result'] = {
            'prediction': "Heart Disease Detected" if prediction == 1 else "Healthy / Low Risk",
            'probability': round(probability * 100, 2),
            'is_high_risk': bool(prediction == 1)
        }

        # 4. Return success and the redirect URL
        return jsonify({"status": "success", "redirect": "/results"})

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

# --- 6. THE AI CHATBOT LOGIC ---

@app.route('/chatbot/message', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({"reply": "Please login first."})

    user_message = request.json.get('message', '').strip().lower()
    
    if 'chat_data' not in session:
        session['chat_data'] = {}
        session['current_step'] = -1 

    # START TRIGGER
    if session['current_step'] == -1:
        if any(word in user_message for word in ["predict", "check", "start", "hi"]):
            session['current_step'] = 0
            return jsonify({"reply": f"Hello! Let's start. {QUESTIONS[0][1]}"})
        return jsonify({"reply": "Type 'Predict' to start."})

    # TRIGGER PREDICTION (This fixes your 'go' issue)
    if user_message == "go" and len(session.get('chat_data', {})) == len(QUESTIONS):
        try:
            ordered_vals = [session['chat_data'][q[0]] for q in QUESTIONS]
            final_features = scaler.transform([ordered_vals])
            
            # Using SVM as mentioned in your screenshot
            prediction = svm_model.predict(final_features)[0]
            probability = svm_model.predict_proba(final_features)[0][1]
            result_text = "Positive for Heart Disease" if prediction == 1 else "Negative / Healthy"

            # Save to Database
            new_pred = HeartPrediction(
                user_id=session['user_id'],
                **session['chat_data'],
                prediction_result=result_text,
                probability=round(probability * 100, 2)
            )
            db.session.add(new_pred)
            db.session.commit()

            # IMPORTANT: Clear session and return the result
            session.pop('chat_data', None)
            session.pop('current_step', None)

            return jsonify({
                "reply": f"Analysis Complete! Result: **{result_text}** (Confidence: {round(probability*100, 2)}%).",
                "finished": True
            })
        except Exception as e:
            return jsonify({"reply": f"Error during analysis: {str(e)}"})

    # DATA COLLECTION
    step = session['current_step']
    field_name = QUESTIONS[step][0]

    try:
        session['chat_data'][field_name] = float(user_message)
        session['current_step'] += 1
        session.modified = True 

        if session['current_step'] < len(QUESTIONS):
            return jsonify({"reply": QUESTIONS[session['current_step']][1]})
        else:
            return jsonify({"reply": "All data collected. Ready to run the AI Diagnostic? (Type 'go' to analyze)"})
            
    except ValueError:
        return jsonify({"reply": f"Please enter a number for {field_name}."})
@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))
@app.route('/history')
def history():
    # Fetch previous predictions for the logged-in user
    user_history = HeartPrediction.query.filter_by(user_id=session.get('user_id')).all()
    return render_template('history.html', history=user_history)

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)