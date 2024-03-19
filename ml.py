import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df.head()

df.isnull().sum()

df.dropna(inplace=False)

df.rename(columns={'Made Donation in March 2007': 'Eligible for donation or not'}, inplace=True)
df

X = df[['Months since Last Donation', 'Number of Donations', 'Total Volume Donated ', 'Months since First Donation']]
y = df['Eligible for donation or not']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc_model = SVC(kernel='linear', random_state=42)

svc_model.fit(X_train_scaled, y_train)

y_pred = svc_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))