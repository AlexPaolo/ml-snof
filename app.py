import pickle
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import json

model_file = 'my_model3.bin'

# Cargar el modelo
with open(model_file, 'rb') as f:
    loaded_model = pickle.load(f)

#X = dv.transform([cliente3])
app = Flask('churn')

@app.route("/")
def inicio():
    return {"mensaje": "API funcionando"}

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    #json_formatted_str = json.dumps(customer, indent=2)
    #print("customer: "+ json_formatted_str)

    #new_df = pd.DataFrame([customer])
    new_df = pd.DataFrame([customer])
    #json_formatted_str2 = json.dumps(new_df, indent=2)
    #print("new_df: "+json_formatted_str2)
    #Tomo la predicción que sea 1
    predictions = loaded_model.predict(new_df)

    result = {
        'prediction': predictions[0],
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

