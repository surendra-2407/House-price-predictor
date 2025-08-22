import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and columns once
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        balcony = int(request.form['balcony'])
        bhk = int(request.form['bhk'])
        location = request.form['location']
        area_type = request.form['area_type']
        availability = request.form['availability']

        user_data = pd.DataFrame(
            [[total_sqft, bath, balcony, bhk, location, area_type, availability]],
            columns=['total_sqft', 'bath', 'balcony', 'bhk', 'location', 'area_type', 'availability']
        )

        user_data = pd.get_dummies(user_data, columns=['location', 'area_type', 'availability'])

        # Align with training columns
        for col in model_columns:
            if col not in user_data.columns:
                user_data[col] = 0

        user_data = user_data[model_columns]
        predicted_price = model.predict(user_data)[0]

        return render_template('index.html',
                               prediction_text=f"Predicted Price: ₹{predicted_price:,.2f} Lakhs")

    except Exception as error_message:
        return render_template('index.html',
                               prediction_text=f"Error: {str(error_message)}")


# 👇 Add this only for local dev
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
