from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# Load trained model
model = joblib.load("/content/best_demand_forecast_model.pkl")

# Allowed options
VALID_STORES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VALID_ITEMS = [15, 28, 13, 18, 25, 45, 38, 22, 36, 8]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_results = None
    error_message = None

    if request.method == "POST":
        try:
            store_id = int(request.form["store_id"])
            item_id = int(request.form["item_id"])
            start_date = request.form["date"]

            # Validation
            if store_id not in VALID_STORES:
                error_message = f"Invalid Store ID! Please select from {VALID_STORES}."
            elif item_id not in VALID_ITEMS:
                error_message = f"Invalid Item ID! Please select from {VALID_ITEMS}."
            else:
                # Convert date
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
                results = []

                # Predict 15 days
                for i in range(15):
                    target_date = start_date + timedelta(days=i)
                    year, month, day_of_week = target_date.year, target_date.month, target_date.weekday()
                    is_weekend = 1 if day_of_week >= 5 else 0

                    features = pd.DataFrame([{
                        "store": store_id,
                        "item": item_id,
                        "year": year,
                        "month": month,
                        "day_of_week": day_of_week,
                        "is_weekend": is_weekend,
                        "prev_sunday_sales": 0  # placeholder
                    }])

                    pred = model.predict(features)[0]
                    results.append((target_date.strftime("%Y-%m-%d"), round(pred, 2)))

                prediction_results = results

        except Exception as e:
            error_message = str(e)

    return render_template(
        "index.html",
        prediction_results=prediction_results,
        error_message=error_message,
        valid_stores=VALID_STORES,
        valid_items=VALID_ITEMS
    )

if __name__ == "__main__":
    app.run(debug=True)
