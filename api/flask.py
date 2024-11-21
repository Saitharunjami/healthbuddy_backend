from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from groq import Groq
import awsgi

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://myhealthbuddy.netlify.app"}})

# --- Diabetes Model ---
diabetes_model = joblib.load('diabetes_model.pkl')

@app.route('/api/diabetes/predict', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()
        input_features = [
            data.get('Pregnancies', 0),
            data.get('Glucose', 0),
            data.get('BloodPressure', 0),
            data.get('SkinThickness', 0),
            data.get('Insulin', 0),
            data.get('BMI', 0.0),
            data.get('DiabetesPedigreeFunction', 0.0),
            data.get('Age', 0)
        ]
        input_features = np.array(input_features).reshape(1, -1)
        prediction = diabetes_model.predict(input_features)[0]
        result = {
            "Prediction": int(prediction),
            "Message": "Diabetes detected" if prediction == 1 else "No diabetes detected"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

# --- Stroke Model ---
stroke_model = joblib.load("stroke_prediction_model_with_smote.pkl")

@app.route("/stroke/predict", methods=["POST"])
def predict_stroke():
    try:
        data = request.get_json()
        sample_features = pd.DataFrame([data])
        sample_features = sample_features[stroke_model.feature_names_in_]
        prediction = stroke_model.predict(sample_features)[0]
        probability = stroke_model.predict_proba(sample_features)[0, 1]
        response = {
            "prediction": "Stroke Predicted" if prediction == 1 else "No Stroke Predicted",
            "probability": f"{probability:.2f}"
        }
        return jsonify(response)
    except KeyError as e:
        return jsonify({"error": f"Missing input field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

# --- Hypertension Model ---
hypertension_model = joblib.load("best_model.pkl")
hypertension_scaler = joblib.load("scaler.pkl")

@app.route("/hypertension/predict", methods=["POST"])
def predict_hypertension():
    try:
        data = request.get_json()
        features = [
            data.get("Age", 0),
            data.get("Sex", 0),
            data.get("HighChol", 0),
            data.get("CholCheck", 0),
            data.get("BMI", 0),
            data.get("Smoker", 0),
            data.get("HeartDiseaseorAttack", 0),
            data.get("PhysActivity", 0),
            data.get("Fruits", 0),
            data.get("Veggies", 0),
            data.get("HvyAlcoholConsump", 0),
            data.get("GenHlth", 0),
            data.get("MentHlth", 0),
            data.get("PhysHlth", 0),
            data.get("DiffWalk", 0),
            data.get("Stroke", 0),
            data.get("Diabetes", 0)
        ]
        scaled_features = hypertension_scaler.transform([features])
        prediction = hypertension_model.predict(scaled_features)[0]
        prediction_label = "Hypertensive" if prediction == 1 else "Not Hypertensive"
        response = {
            "prediction": prediction_label,
            "input": data
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Heart Disease Model --- 
import joblib
import pandas as pd

# Load the trained Heart Disease model and scaler
heart_disease_model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

# Define the expected feature names based on the heart disease dataset
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 
    'fbs', 'restecg', 'thalach', 'exang', 
    'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/heart_disease/predict', methods=['POST'])
def predict_heart_disease():
    try:
        # Get JSON data from the request
        data = request.json
        #print("Received data:", data)

        # Format input data according to the expected features
        input_data = {feature: [data.get(feature, 0)] for feature in FEATURE_NAMES}
        input_df = pd.DataFrame.from_dict(input_data)

        # Log formatted DataFrame
        #print("Formatted input DataFrame:", input_df)

        # Scale the input data
        scaled_data = scaler.transform(input_df)

        # Predict with the Heart Disease model
        prediction = heart_disease_model.predict(scaled_data)[0]
        probability = heart_disease_model.predict_proba(scaled_data)[:, 1][0]

        # Send response back to frontend
        response = {
            "Message": "Heart disease detected" if prediction == 1 else "No heart disease detected",
            "Probability": round(probability * 100, 2)  # Convert to percentage
        }
        print("Response:", response)
        return jsonify(response)

    except Exception as e:
        # Log and return error
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 400

    
    

# --- Mental Health Model ---
mental_health_model = joblib.load("mental_health_model.pkl")
MENTAL_HEALTH_FEATURE_NAMES = [
    "Age", "Gender", "self_employed", "family_history", "work_interfere",
    "no_employees", "remote_work", "tech_company", "benefits", "care_options",
    "wellness_program", "seek_help", "anonymity", "leave",
    "mental_health_consequence", "phys_health_consequence", "coworkers",
    "supervisor", "mental_health_interview", "phys_health_interview",
    "mental_vs_physical", "obs_consequence"
]

@app.route('/mental_health/predict', methods=['POST'])
def predict_mental_health():
    try:
        data = request.get_json()
        input_data = {feature: [data[feature]] for feature in MENTAL_HEALTH_FEATURE_NAMES}
        input_df = pd.DataFrame.from_dict(input_data)
        prediction = mental_health_model.predict(input_df)
        probability = mental_health_model.predict_proba(input_df)[:, 1]
        response = {
            "Message": "Treatment needed" if prediction[0] == 1 else "No treatment needed",
            "Probability": float(probability[0])
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400



# --- Chatbot Model (Groq) ---
chat_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
history = []

def get_chatbot_response(user_input):
    chat_completion = chat_client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

@app.route('/chat', methods=['POST'])
def chat():
    global history
    user_input = request.json.get('user_input')
    if user_input:
        try:
            bot_response = get_chatbot_response(user_input)
            history.append({"user": user_input, "bot": bot_response})
            return jsonify({"bot_response": bot_response, "history": history})
        except Exception as e:
            return jsonify({"error": "Failed to get response from the chatbot"}), 500
    return jsonify({"error": "No input provided"}), 400



# --- Skin Disease Endpoint ---

# Skin Disease Model
MODEL_PATH = "skin_disease_model.h5"
skin_disease_model = load_model(MODEL_PATH)


# Class labels for skin disease
SKIN_DISEASE_LABELS = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections",
]


@app.route('/skin_disease/predict', methods=['POST'])
def predict_skin_disease():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the file temporarily
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        # Preprocess the image
        image = load_img(file_path, target_size=(224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict
        predictions = skin_disease_model.predict(image)
        predicted_class = SKIN_DISEASE_LABELS[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Remove the temporary file
        os.remove(file_path)

        # Return the response
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    

# --- Obesity Model ---
model_path = "obesity_model.pkl"
obesity_model, obesity_preprocessor, obesity_label_encoder = joblib.load(model_path)

OBESITY_FEATURE_NAMES = [
    'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 
    'NCP', 'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 
    'FAF', 'TUE', 'CAEC', 'MTRANS'
]

@app.route('/obesity/predict', methods=['POST'])
def predict_obesity():
    try:
        data = request.get_json()
        app.logger.debug("Received data: %s", data)

        # Normalize keys to match FEATURE_NAMES
        key_mapping = {key.lower(): key for key in OBESITY_FEATURE_NAMES}
        normalized_data = {
            key_mapping[key.lower()]: value for key, value in data.items() if key.lower() in key_mapping
        }
        app.logger.debug("Normalized data: %s", normalized_data)

        # Validate input
        missing_features = [feature for feature in OBESITY_FEATURE_NAMES if feature not in normalized_data]
        if missing_features:
            return jsonify({"error": f"Missing input fields: {missing_features}"}), 400

        # Prepare input data for prediction
        input_data = {feature: [normalized_data[feature]] for feature in OBESITY_FEATURE_NAMES}
        input_df = pd.DataFrame.from_dict(input_data)
        input_preprocessed = obesity_preprocessor.transform(input_df)

        # Make prediction
        prediction = obesity_model.predict(input_preprocessed)
        predicted_label = obesity_label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "predicted_label": predicted_label,
            "message": f"The predicted obesity class is: {predicted_label}"
        })
    except Exception as e:
        app.logger.error("Error: %s", str(e))
        return jsonify({"error": str(e)}), 500

# --- Sleep Apnea Model ---
model_path_sleep = "sleep_apnea_model.pkl"
sleep_apnea_model, sleep_apnea_label_encoder = joblib.load(model_path_sleep)

FEATURE_NAMES_SLEEP = [
    "Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level", "BMI Category", "Heart Rate",
    "Daily Steps", "Systolic BP", "Diastolic BP"
]

@app.route('/sleep_apnea/predict', methods=['POST'])
def predict_sleep_apnea():
    try:
        data = request.get_json()
        missing_features = [feature for feature in FEATURE_NAMES_SLEEP if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing input fields: {missing_features}"}), 400
        input_df = pd.DataFrame([data])
        predicted_class_encoded = sleep_apnea_model.predict(input_df)[0]
        predicted_class = sleep_apnea_label_encoder.inverse_transform([predicted_class_encoded])[0]
        return jsonify({
            "predicted_class": predicted_class,
            "message": f"The predicted sleep disorder is: {predicted_class}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
