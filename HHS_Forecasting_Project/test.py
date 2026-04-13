import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("data1/data.csv1")

# Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df = df.drop_duplicates(subset='Date')
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df = df.asfreq('D')
df = df.interpolate()

# Convert to numeric
df['Children in HHS Care'] = pd.to_numeric(df['Children in HHS Care'], errors='coerce')
df['Children transferred out of CBP custody'] = pd.to_numeric(df['Children transferred out of CBP custody'], errors='coerce')
df['Children discharged from HHS Care'] = pd.to_numeric(df['Children discharged from HHS Care'], errors='coerce')

# Feature Engineering
df['lag_1'] = df['Children in HHS Care'].shift(1)
df['lag_7'] = df['Children in HHS Care'].shift(7)
df['rolling_mean_7'] = df['Children in HHS Care'].rolling(7).mean()
df['net_flow'] = df['Children transferred out of CBP custody'] - df['Children discharged from HHS Care']
df['day_of_week'] = df.index.dayofweek

df = df.dropna()


#  MODEL BUILDING


# Features & Target
X = df.drop(columns=['Children in HHS Care'])
y = df['Children in HHS Care']

# Time-based split
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
preds = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)

print("Model Results ✅")
print("MAE:", mae)
print("RMSE:", rmse)