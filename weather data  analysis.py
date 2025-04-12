import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
df = pd.read_csv('C:/Users/Lenovo/Desktop/Sudeep/weather.csv')

# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 3: Data Visualization
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()

# Step 4: Feature Engineering - Create 'Date' column for 366 days
start_date = '2024-01-01'  # Start of a leap year
df['Date'] = pd.date_range(start=start_date, periods=366, freq='D')  # Exactly 366 days
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Step 5: Data Analysis
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

# Step 6: Data Visualization (Part 2)
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature (°C)')
plt.title('Monthly Average Max Temperature (2024)')
plt.grid(True)
plt.show()

# Step 7: Advanced Analysis (predict Rainfall)
X = df[['MinTemp', 'MaxTemp', 'Humidity9am', 'Pressure9am']]  # Added more features
y = df['Rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')

# Step 8: Conclusions and Insights
highest_temp_month = monthly_avg_max_temp.idxmax()
lowest_temp_month = monthly_avg_max_temp.idxmin()
print(f'Month with highest average MaxTemp: {highest_temp_month} (1=Jan, 12=Dec)')
print(f'Month with lowest average MaxTemp: {lowest_temp_month} (1=Jan, 12=Dec)')

# Optional: Rainfall Insights
monthly_avg_rainfall = df.groupby('Month')['Rainfall'].mean()
highest_rainfall_month = monthly_avg_rainfall.idxmax()
lowest_rainfall_month = monthly_avg_rainfall.idxmin()
print(f'Month with highest average Rainfall: {highest_rainfall_month} (1=Jan, 12=Dec)')
print(f'Month with lowest average Rainfall: {lowest_rainfall_month} (1=Jan, 12=Dec)')