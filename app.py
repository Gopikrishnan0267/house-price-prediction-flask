from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    bedrooms = request.form['bedrooms']
    bathrooms = request.form['bathrooms']
    floors = request.form['floors']
    yr_built = request.form['yr_built']

    # Convert input values into numbers
    arr = np.array(
        [bedrooms, bathrooms, floors, yr_built],
        dtype=np.float64
    )

    # Predict house price
    prediction = model.predict([arr])

    # Get prediction value
    result = prediction[0]

    return render_template(
        'index.html',
        data=result
    )


if __name__ == '__main__':
    app.run(debug=True)