# Define the lung cancer risk assessment function
def lung_cancer_risk_assessment(data):
    score = 0
    
    # Assign weights to the primary features
    primary_weights = {
        'SMOKING': 3,
        'YELLOW_FINGERS': 2,
        'COUGHING': 3,
        'SHORTNESS OF BREATH': 3,
        'WHEEZING': 2,
        'CHEST PAIN': 2,
        'SWALLOWING DIFFICULTY': 2
    }
    
    # Assign lower weights to the secondary features
    secondary_weights = {
        'AGE': 1,
        'ANXIETY': 1,
        'PEER_PRESSURE': 1,
        'CHRONIC DISEASE': 1,
        'FATIGUE': 1,
        'ALLERGY': 1,
        'ALCOHOL CONSUMING': 1
    }
    
    # Calculate the score from primary features
    for feature, weight in primary_weights.items():
        score += weight * data.get(feature, 0)
    
    # Calculate the score from secondary features
    for feature, weight in secondary_weights.items():
        score += weight * data.get(feature, 0)
    
    # Determine lung cancer likelihood based on score
    if score >= 15:
        return "High likelihood of lung cancer"
    elif score >= 10:
        return "Moderate likelihood of lung cancer"
    else:
        return "Low likelihood of lung cancer"


from flask import Flask, request, jsonify, render_template, session
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load the trained model
model = joblib.load('best_hyperparameter_tuned_model.pkl')

# The list of questions corresponding to each key feature
questions = [
    "What is your gender? (Male: 1, Female: 0)",
    "Are you older than 65?",
    "Do you smoke?",
    "Do you have yellow fingers?",
    "Do you experience anxiety?",
    "Are you under peer pressure?",
    "Do you have a chronic disease?",
    "Do you experience fatigue?",
    "Do you have any allergies?",
    "Do you wheeze?",
    "Do you consume alcohol?",
    "Do you have a cough?",
    "Do you experience shortness of breath?",
    "Do you have difficulty swallowing?",
    "Do you have chest pain?"
]

@app.route('/')
def home():
    session['responses'] = []
    session['question_index'] = 0
    return render_template('main.html', question=questions[0])

@app.route('/next_question', methods=['POST'])
def next_question():
    data = request.get_json()
    response = data.get('response')  # Use .get() to avoid KeyError
    
    # Store the response
    session['responses'].append(int(response))
    session['question_index'] += 1
    
    # If all questions have been answered
    if session['question_index'] >= len(questions):
        # Create a dictionary of responses with corresponding feature names
        feature_names = [
            "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", 
            "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY", 
            "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", 
            "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
        ]
        responses = dict(zip(feature_names, session['responses']))
        
        # Calculate risk score using the lung_cancer_risk_assessment function
        risk_assessment = lung_cancer_risk_assessment(responses)
        
        # Use the responses for model prediction
        features = np.array(session['responses']).reshape(1, -1)
        prediction = model.predict(features)[0]
        model_result = "Lung Cancer Detected" if prediction == 1 else "No Lung Cancer Detected"
        
        # Combine the model prediction with the risk assessment
        final_result = f"{model_result}. Risk Assessment: {risk_assessment}"
        return jsonify(result=final_result)
    
    # Ask the next question
    next_question = questions[session['question_index']]
    return jsonify(question=next_question)

if __name__ == "__main__":
    app.run(debug=True)
