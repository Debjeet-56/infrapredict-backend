from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

FILE_IDS = {
    "explainer_labour"    : "14Fqsegh1SqyvEx49NPJ67ixZQ57mcPHN",
    "explainer_duration"  : "1R76i6aTLVWxqyJiY8ruBQPmFPVhTOsQ_",
    "explainer_logistics" : "1GGl_3W4_E5SwSchxEqqX4KCNf7LO0UuQ",
    "explainer_material"  : "1CQ148FRRTVYNhdszpbYKo9mLAAEmwABR",
    "explainer_budget"    : "15fN4MujY8OKWMbYGNOPk8HTn6gsYGA1J",
    "model_labour"        : "1uODOi1kIPDplAvskS3YEnwMpkbFXXKjF",
    "model_duration"      : "1TXuWJxvt2VdWeBpKt-c8M2P5v22pV1hw",
    "model_logistics"     : "13lZPTBmCgKCE6WxjKpk8sRPZ0wEReG-j",
    "model_material"      : "1CjfPEn9nqeoDYSx8r4meAiD10sz11btX",
    "model_budget"        : "1IVzK0-CXZcnVZXCi2XNAsGCu1BJ1Ex-O",
    "X_test"              : "1vTt2UEqf86iGgPa3q-DjlK6a3VwmQn7t",
    "X_shap"              : "15dapjcMZw9JHAtg1SaJ6nGfL7VnZv7k8",
    "shap_duration"       : "1hP6hgo7_ihVrPu9l3243xM5a5DIpnhr8",
    "shap_logistics"      : "1dv-ypPkT8uO_8FLX_XMauBCXxyzIJ76t",
    "shap_material"       : "1W0dca5Lx2XFjgXwkUUvv8sw3bxiQrAjP",
    "shap_budget"         : "1Tnpxww4VILokdnBgzPN8Tw_TBZ5Drq-b",
    "shap_labour"         : "1R5cYOPpj2B7yto484qENhoh8YWJApDeF",
    "encoders"            : "1grGIUpih8FZuT4CpaDk1oDRmCbplj67F",
}

os.makedirs('models', exist_ok=True)

def download_file(file_id, dest_path):
    print(f"⏳ Downloading {dest_path}...")
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(url, stream=True)

    # Handle large file warning page
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
        response = session.get(url, stream=True)

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"✅ {dest_path} done!")

def download_all():
    for name, file_id in FILE_IDS.items():
        path = f'models/{name}.pkl'
        if not os.path.exists(path):
            download_file(file_id, path)
        else:
            print(f"✅ {name} already exists, skipping...")

print("⏳ Downloading all files from Google Drive...")
download_all()
print("✅ All files ready!")

# Load models
model_duration  = joblib.load('models/model_duration.pkl')
model_budget    = joblib.load('models/model_budget.pkl')
model_material  = joblib.load('models/model_material.pkl')
model_logistics = joblib.load('models/model_logistics.pkl')
model_labour    = joblib.load('models/model_labour.pkl')

# Load explainers
explainer_duration  = joblib.load('models/explainer_duration.pkl')
explainer_budget    = joblib.load('models/explainer_budget.pkl')
explainer_material  = joblib.load('models/explainer_material.pkl')
explainer_logistics = joblib.load('models/explainer_logistics.pkl')
explainer_labour    = joblib.load('models/explainer_labour.pkl')

# Load encoders
encoders = joblib.load('models/encoders.pkl')

print("✅ All models and explainers loaded!")

FEATURE_NAMES = [
    'Project_Type', 'Terrain_Type', 'Region', 'Material_Cost_Volatility',
    'Estimated_Duration_Weeks', 'Estimated_Budget_Lakhs',
    'Team_Size', 'Num_Subcontractors', 'Permits_Required', 'Risk_Score'
]

def get_hotspots(explainer, features):
    shap_vals = explainer.shap_values(features)[0]
    total = sum(abs(s) for s in shap_vals)
    hotspots = []
    for name, val in zip(FEATURE_NAMES, shap_vals):
        pct = round(abs(val) / total * 100, 1)
        hotspots.append({
            "feature"      : name,
            "contribution" : pct,
            "direction"    : "increases risk" if val > 0 else "reduces risk",
            "level"        : "HIGH" if pct > 20 else "MEDIUM" if pct > 10 else "LOW"
        })
    return sorted(hotspots, key=lambda x: x['contribution'], reverse=True)[:5]

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "✅ InfraPredictAI API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        text_columns = ['Project_Type', 'Terrain_Type', 'Region', 'Material_Cost_Volatility']
        for col in text_columns:
            if col in data:
                data[col] = int(encoders[col].transform([data[col]])[0])

        features = np.array([[
            data['Project_Type'],
            data['Terrain_Type'],
            data['Region'],
            data['Material_Cost_Volatility'],
            data['Estimated_Duration_Weeks'],
            data['Estimated_Budget_Lakhs'],
            data['Team_Size'],
            data['Num_Subcontractors'],
            data['Permits_Required'],
            data['Risk_Score'],
        ]])

        return jsonify({
            "predicted_duration_weeks"   : round(float(model_duration.predict(features)[0]), 2),
            "predicted_budget_lakhs"     : round(float(model_budget.predict(features)[0]), 2),
            "predicted_material_change"  : round(float(model_material.predict(features)[0]), 2),
            "predicted_logistics_change" : round(float(model_logistics.predict(features)[0]), 2),
            "predicted_labour_change"    : round(float(model_labour.predict(features)[0]), 2),
            "hotspots": {
                "duration"  : get_hotspots(explainer_duration,  features),
                "budget"    : get_hotspots(explainer_budget,    features),
                "material"  : get_hotspots(explainer_material,  features),
                "logistics" : get_hotspots(explainer_logistics, features),
                "labour"    : get_hotspots(explainer_labour,    features),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
