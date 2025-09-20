# %% [markdown]
# Import the neccessary libraries
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# %% [markdown]
# Data Loading

# %%
print("Loading advertising data")
df = pd.read_csv("advertising.csv")
print(df.head())
print("Data Loaded Sucessfully!")


# %% [markdown]
# Data Preprocessing and Feature Engineering

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing values")
print(df_missing)

# Check for duplicates
df_duplicates = df.duplicated().sum()
print("Duplicated values")
print(df_duplicates)

# Rename the columns for clarity and consistency
df.rename(columns={
    "TV":"tv",
    "Radio":"radio",
    "Newspaper":"newspaper",
    "Sales":"sales"
},inplace=True)
print(df.info())

# Define the features (X) and target (y)
features = ["tv"]
target = ["sales"]

X = df[features]
y = df[target]

print("Shape of features (X):",X.shape)
print("Shape of target (y):",y.shape)

# %% [markdown]
# Data Visualization before training

# %%
plt.scatter(X,y)
plt.title("Sales vs TV")
plt.xlabel("Tv")
plt.ylabel("Sales")
plt.grid()
plt.show()

# %% [markdown]
# Data Splitting

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("Number of sample in training set:", len(X_train))
print("Number of sample in the testing set:",len(X_test))

# %% [markdown]
# Model Training

# %%
print("Training the Linear Regression model.........")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model Training complete!")

# %% [markdown]
# Model Evaluation

# %%
print("Evaluating the model")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared : {r2:.2f}")

# You can also look at the coefficients to understand the model's equation
print("Model Coefficients")
print("coefficients: ", model.coef_)
print("intercept: ", model.intercept_)

# %% [markdown]
# Visualization of Results

# %%
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.show()

# %% [markdown]
# Making a New Prediction

# %%
new_data = pd.DataFrame({
    "tv":[200]
})

predicted_sales = model.predict(new_data)
print(f"Predicted Sales: {predicted_sales}")


