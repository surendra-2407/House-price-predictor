import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = "Bengaluru_House_Data (1).csv"
df = pd.read_csv(data)
df.drop(columns=['society'], errors="ignore", inplace=True)
print(df.isnull().sum())

bhk_list = [] 
for val in df['size']:
    if pd.notnull(val):
        num = str(val).split(' ')[0]
        bhk_list.append(int(num))
    else:
        bhk_list.append(None)
df['BHK'] = bhk_list  
df.drop(['size'], errors="ignore", inplace=True)

def sqft_to_number(x):
    try:
        return float(x)
    except:
        vals = str(x).split('-')
        if len(vals) == 2:
            return (float(vals[0]) + float(vals[1])) / 2
        return None

df['total_sqft'] = df['total_sqft'].apply(sqft_to_number)

df.dropna(inplace=True)

plt.figure(figsize=(8, 4))
sns.boxplot(x=df["price"])
plt.title("Price Distribution (with Outliers)")
plt.show()

df["price"].hist(bins=30, figsize=(8, 4))
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Price Histogram")
plt.show()

plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.drop(['price'], axis=1)
y = df['price']

X = pd.get_dummies(X, columns=['location','area_type', 'availability'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)


model = RandomForestRegressor(n_estimators=200, random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print("Mean Absolute Error:",mae)
print("Root Mean Squared Error:",rmse)
print("R2 Score",r2)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted Prices")
plt.show()

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))