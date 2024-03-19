from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

app = Flask(__name__)

# Load the trained model
data = pd.read_csv("data.csv")
data.dropna(inplace=True)
selected_features = ['Months since Last Donation', 'Number of Donations', 'Total Volume Donated ', 'Months since First Donation']
x = data[selected_features]
y = data["Eligible for donation or not"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
model = SVC()
model.fit(x_train, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    MonthssinceLastDonation = float(request.form['Months since Last Donation'])
    NumberofDonations = float(request.form['Number of Donations'])
    TotalVolumeDonated = float(request.form['Total Volume Donated '])
    MonthssinceFirstDonation = float(request.form['Months since First Donation'])
        
    user_input = np.array([[MonthssinceLastDonation, NumberofDonations, TotalVolumeDonated, MonthssinceFirstDonation]])
        
    # Make prediction
    predicted_eligibility = model.predict(user_input)
        
    # Redirect to result.html with the predicted eligibility value
    return render_template('result.html', eligibility=predicted_eligibility[0])

if __name__ == '__main__':
    app.run(debug=True)
