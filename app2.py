from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

# Example training data
x_train = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]])
y_train = np.array([1,0,1])

# Initialize and train the RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

app = Flask(__name__)
with open('C:/Users/sures/Downloads/covid/covid2.pkl','rb') as f:
    model=pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:
            # Handle empty inputs
            depth = float(request.form['depth']) if request.form['depth'] else 0.0
            deaths = float(request.form['deaths']) if request.form['deaths'] else 0.0
            recovered = float(request.form['recovered']) if request.form['recovered'] else 0.0
            value = float(request.form['value']) if request.form['value'] else 0.0

            input_data = np.array([[depth, deaths, recovered, value]])

            # Make a prediction using the trained model
            prediction = clf.predict(input_data)[0]
            return render_template('index.html', prediction=prediction)

        except ValueError as e:
            return render_template('index.html', prediction="Invalid input: " + str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)