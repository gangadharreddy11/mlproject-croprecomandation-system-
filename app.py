# from flask import Flask, render_template, request
# import numpy as np
# import joblib
# import os
# app = Flask(__name__)

# # Build correct model file path
# model_path = os.path.join(
#     os.path.dirname(__file__),
#     "dt_model_joblib.pkl"
# )
# # Load the trained model
# model = joblib.load(model_path)

# @app.route("/")
# def home():
#     # Initial page load
#     return render_template("index.html", prediction_text="")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get data from form
#         N          = float(request.form["N"])
#         P          = float(request.form["P"])
#         K          = float(request.form["K"])
#         temperature= float(request.form["temperature"])
#         humidity   = float(request.form["humidity"])
#         ph         = float(request.form["ph"])
#         rainfall   = float(request.form["rainfall"])

#         # Prepare features as 2D array for model
#         features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

#         # Predict
#         prediction = model.predict(features)[0]

#         result_text = f"Recommended crop: {prediction}"

#     except Exception as e:
#         result_text = f"Error: {e}"

#     # Render same page with result
#     return render_template("index.html", prediction_text=result_text)

# if __name__ == "__main__":
#     app.run(debug=True)


# ================== adtional add the crop suggetion ==============

from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# ---- Load model ----
model_path = os.path.join(
    os.path.dirname(__file__),
    "dt_model_joblib.pkl"
)
model = joblib.load(model_path)

# ---- Crop nutrients & vitamins database ----
crop_info = {
    "rice": {
        "nutrients": "Carbohydrates, Fiber, Small Protein",
        "vitamins": "Vitamin B1, Vitamin B3, Vitamin D"
    },
    "maize": {
        "nutrients": "Carbohydrates, Fiber, Proteins",
        "vitamins": "Vitamin A, Vitamin B, Vitamin E"
    },
    "banana": {
        "nutrients": "Carbohydrates, Potassium, Fiber",
        "vitamins": "Vitamin B6, Vitamin C"
    },
    "apple": {
        "nutrients": "Fiber, Natural Sugars",
        "vitamins": "Vitamin C, Vitamin K"
    },
    "chickpea": {
        "nutrients": "Proteins, Fiber, Iron",
        "vitamins": "Vitamin B9 (Folate)"
    }
}

# ---- Home route ----
@app.route("/")
def home():
    return render_template("index.html", prediction_text="", nutrients="", vitamins="")

# ---- Prediction route ----
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Prepare data for model
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict crop
        prediction = model.predict(features)[0]

        # Get nutrients + vitamins
        info = crop_info.get(prediction, {
            "nutrients": "Information not available",
            "vitamins": "Information not available"
        })

        nutrients = info["nutrients"]
        vitamins = info["vitamins"]

        return render_template(
            "index.html",
            prediction_text=f"Recommended crop: {prediction}",
            nutrients=nutrients,
            vitamins=vitamins
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}",
            nutrients="",
            vitamins=""
        )

if __name__ == "__main__":
    app.run(debug=True)
