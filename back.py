from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Utility functions
def encode_symptoms(symptoms):
    all_symptoms = ['itching', 'fatigue', 'vomiting', 'weight_loss', 'lethargy', 
                    'cough', 'high_fever', 'nausea', 'loss_of_appetite']
    return [1 if sym in symptoms else 0 for sym in all_symptoms]

def categorize_bp(bp_input):
    try:
        parts = bp_input.split('/')
        if len(parts) == 2:
            systolic = int(parts[0].strip())
            diastolic = int(parts[1].strip())
        else:
            systolic = int(bp_input.strip())
            diastolic = 0
    except Exception:
        return 'Unknown'
    if systolic < 90 or diastolic < 60:
        return 'Low'
    elif 90 <= systolic <= 120 and 60 <= diastolic <= 80:
        return 'Normal'
    elif systolic > 120 or diastolic > 80:
        return 'High'
    else:
        return 'Unknown'

# Load dataset
df = pd.read_csv("detailed_exercise_disease_dataset.csv")
df.fillna('', inplace=True)
df['combined_text'] = df['Disease'] + ' ' + df['Past_History'] + ' ' + df['BP_Level']

text_feature = 'combined_text'
numeric_features = ['Age', 'Heart_Beats', 'Oxygen_Level']
output_labels = ['Medicine_1', 'Medicine_2', 'Medicine_3', 'Medicine_4',
                 'Exercise 1', 'Exercise 2', 'Exercise 3']
regression_target = 'Recovery_Days'

preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(), text_feature),
    ('num', StandardScaler(), numeric_features)
])

X = df[[text_feature] + numeric_features]
y_classification = df[output_labels]
y_regression = df[regression_target]

clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
])
clf_pipeline.fit(X, y_classification)

reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
reg_pipeline.fit(X, y_regression)

disease_list = sorted(df['Disease'].dropna().unique())

# Load disease prediction model
model = pickle.load(open('pred.pkl', 'rb'))
symptom_encoders = pickle.load(open('symptom_encoders.pkl', 'rb'))
disease_encoder = pickle.load(open('disease_encoder.pkl', 'rb'))

# In-memory user store
users = {}

@app.before_request
def before_request():
    session.permanent = True

@app.route('/')
def root():
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('signup.html', error='Username already exists')
        users[username] = password
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) == password:
            session['user'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    symptoms_list = sorted(set.union(*(set(le.classes_) for le in symptom_encoders.values())))
    return render_template('home.html', symptoms=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms')
    if not symptoms or len(symptoms) != 6:
        return jsonify({'error': 'Exactly 6 symptoms are required'}), 400
    try:
        encoded = []
        for i, symptom in enumerate(symptoms):
            col = f'Symptom_{i+1}'
            le = symptom_encoders.get(col)
            if le and symptom in le.classes_:
                encoded.append(le.transform([symptom])[0])
            else:
                return jsonify({'error': f'Invalid symptom: {symptom}'}), 400
        prediction = model.predict([encoded])[0]
        disease = disease_encoder.inverse_transform([prediction])[0]
        return jsonify({'disease': disease})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/treatment', methods=["GET", "POST"])
def treatment():
    if 'user' not in session:
        return redirect(url_for('login'))

    selected_disease = request.args.get('disease', '')
    if request.method == "POST":
        try:
            age = float(request.form['age'])
            disease = request.form['disease']
            past_history = request.form.get('past_history', '')
            raw_bp = request.form['bp']
            heart_beats = float(request.form['heart_beats'])
            oxygen_level = float(request.form['oxygen_level'])

            bp_category = categorize_bp(raw_bp)
            combined_text = disease + ' ' + past_history + ' ' + bp_category

            input_df = pd.DataFrame({
                'combined_text': [combined_text],
                'Age': [age],
                'Heart_Beats': [heart_beats],
                'Oxygen_Level': [oxygen_level]
            })

            pred_class = clf_pipeline.predict(input_df)[0]
            pred_reg = reg_pipeline.predict(input_df)[0]

            medicines_raw = pred_class[:4]
            exercises_raw = pred_class[4:7]

            medicines = [med for med in medicines_raw if isinstance(med, str) and med.strip() and med.lower() != 'nan']
            exercises = [ex for ex in exercises_raw if isinstance(ex, str) and ex.strip() and ex.lower() != 'nan']

            treatment_plan = {
                "BP_Category": bp_category,
                "Recovery_Days": round(pred_reg, 2),
                "Medicines": medicines,
                "Exercises": exercises
            }

            session['treatment_plan'] = treatment_plan
            return redirect(url_for('result'))

        except Exception as e:
            return render_template("index.html", error=str(e), disease=selected_disease)

    return render_template("index.html", disease=selected_disease)

@app.route("/result")
def result():
    treatment_plan = session.get('treatment_plan', None)
    if treatment_plan is None:
        return redirect(url_for('treatment'))
    return render_template("result.html", treatment=treatment_plan)

@app.route("/diseases")
def diseases():
    return render_template("lists.html", diseases=disease_list)

if __name__ == "__main__":
    app.run(debug=True)