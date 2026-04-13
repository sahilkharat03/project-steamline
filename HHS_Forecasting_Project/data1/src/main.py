from preprocessing import load_and_clean_data
from feature_engineering import create_features
from model import train_model 

import matplotlib.pyplot as plt

df = load_and_clean_data("../data.csv1")
df = create_features(df)

y_test, preds = train_model(df)

plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, preds, label="Predicted")

plt.legend()
plt.show()