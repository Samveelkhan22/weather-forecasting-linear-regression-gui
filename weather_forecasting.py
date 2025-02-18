import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import simpledialog, messagebox
import warnings

warnings.filterwarnings("ignore")

# Step 1: Load CSV File
file_path = input("Enter the path to your CSV file: ")
df = pd.read_csv(file_path, parse_dates=['Date'])
df.columns = ['ds', 'y']
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df.dropna(inplace=True)

# Step 2: Feature Engineering for Linear Regression
df['timestamp'] = (df['ds'] - df['ds'].min()).dt.days

X = df[['timestamp']]
y = df['y']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance:\nMSE: {mse:.4f}\nR2 Score: {r2:.4f}")

# Step 6: Get User Date from GUI
root = tk.Tk()
root.withdraw()
user_date_str = simpledialog.askstring("Input", "Enter a date (YYYY-MM-DD) to forecast temperature:")
user_date = pd.to_datetime(user_date_str)
user_timestamp = (user_date - df['ds'].min()).days

# Step 7: Predict Temperature for User Date
predicted_temp = model.predict([[user_timestamp]])[0]

# Step 8: Plot Graph
plt.figure(figsize=(12, 5))
plt.plot(df['ds'], df['y'], label="Historical Temperatures", linewidth=2)
plt.scatter(user_date, predicted_temp, color='red', 
            label=f"Predicted Temp: {predicted_temp:.2f}°C", s=100)
plt.axvline(user_date, color='purple', linestyle='--',
            label=f"Selected Date: {user_date.strftime('%Y-%m-%d')}")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title(f"Linear Regression Forecast for {user_date.strftime('%Y-%m-%d')}")
plt.legend()
plt.grid(True)
plt.pause(0.1)  # Show warning before closing graph

# Step 9: Show Temperature Warning via GUI
if predicted_temp < 5:
    messagebox.showwarning("Temperature Alert", 
                           f"Too Cold: {predicted_temp:.2f}°C on {user_date.strftime('%Y-%m-%d')}")
elif predicted_temp > 35:
    messagebox.showwarning("Temperature Alert", 
                           f"Too Hot: {predicted_temp:.2f}°C on {user_date.strftime('%Y-%m-%d')}")
else:
    messagebox.showinfo("Forecast Result", 
                        f"Predicted Temp: {predicted_temp:.2f}°C on {user_date.strftime('%Y-%m-%d')}")

plt.show()


