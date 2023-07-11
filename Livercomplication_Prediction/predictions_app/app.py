from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
app.debug = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Loading the saved model
lr = joblib.load('/Users/dylanskalman/Downloads/Data_Science_Projects/Hepatitis C Prediction/predictions_app/model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    alb = float(request.form.get('alb'))
    alp = float(request.form.get('alp'))
    alt = float(request.form.get('alt'))
    ast = float(request.form.get('ast'))
    bil = float(request.form.get('bil'))
    che = float(request.form.get('che'))
    chol = float(request.form.get('chol'))
    ggt = float(request.form.get('ggt'))
    prot = float(request.form.get('prot'))

    # Preprocessing on the input data
    input_data = [[age, sex, alb, alp, alt, ast, bil, che, chol, ggt, prot ]]

    # Making predictions using the loaded model
    prediction = lr.predict(input_data)

    # Return the prediction result
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run()

