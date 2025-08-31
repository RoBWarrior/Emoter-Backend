# Import necessary libraries from transformers and torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the path to your downloaded Hugging Face model
MODEL_PATH = "./my_trained_model" # IMPORTANT: Make sure this path is correct

# Set up the device (GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load the new model and tokenizer ---
# This is the replacement for your joblib.load() calls
print("Loading Hugging Face model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval() # Set the model to evaluation mode (important for inference)
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit or handle the error appropriately if the model can't be loaded
    exit()


# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

# (Optional but Recommended) Map model's output IDs to human-readable labels
# You need to adjust these labels based on what your model was trained to predict.
# Map output IDs to your emotion labels
id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

@app.route("/")
def home():
    return "Hugging Face Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # ✅ --- New Prediction Logic ---
    try:
        # 1. Tokenize the input text and move to the correct device
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 2. Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 3. Get the predicted class ID and its label
        predicted_class_id = logits.argmax().item()
        prediction = id2label.get(predicted_class_id, "Unknown") # Use the map here

        # Return a more detailed response
        return jsonify({
            "prediction": prediction,
            "prediction_id": predicted_class_id
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Failed to process the request."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
