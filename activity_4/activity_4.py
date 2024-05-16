import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plot
from sklearn import metrics
from statsmodels import api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Define column headers
headers = [
    "os_type",
    "data_bytes",
    "time"
]

# Read dataset
dataset = pd.read_csv("act4.csv", header=None, names=headers)

# Drop the first row if it contains column names
dataset.drop(index=dataset.index[0], axis="index", inplace=True)

# Convert categorical variable os_type to numerical
dataset["os_type"] = dataset["os_type"].astype("category").cat.codes

# Convert dataset to float (this line seems redundant as it doesn't modify dataset in place)
dataset.astype(float)

# Define feature variables (X) and dependent variable (Y)
feature_var = dataset.drop(columns="time")
dep_var = dataset["time"]

# Fit linear regression model
model = LinearRegression()
model.fit(feature_var, dep_var)

# Predict dependent variable
dep_predict = model.predict(feature_var)

# Plot actual vs. predicted values
plot.figure(figsize=(12, 5))
plot.scatter(dep_var, dep_predict)
plot.xlabel("Actual Time")
plot.ylabel("Predicted Time")
plot.savefig("scatterplot.png")

# Calculate R-squared and save to summary.txt
r2_score = metrics.r2_score(dep_var, dep_predict)
with open("summary.txt", "w") as file:
    file.write(str(float(r2_score)))

# Get coefficients and intercept
coeff = model.coef_
intercept = model.intercept_

# Convert dependent variable to float
dep_var = dep_var.astype(float)

# Calculate residuals
residuals = dep_var.sub(dep_predict)

# Plot Q-Q plot of residuals
sm.qqplot(residuals, line="45")
plot.title("Q-Q Plot of Residuals")
plot.savefig("qqplot_residuals.png")

# Plot residuals vs. fitted values to check homoscedasticity
plot.scatter(dep_predict, residuals)
plot.xlabel("Fitted values")
plot.ylabel("Residuals")
plot.title("Residuals vs Fitted Values")
plot.savefig("homoscedasticity.png")

# Calculate Variance Inflation Factor (VIF) to check multicollinearity
vif = pd.DataFrame()
vif["feature"] = feature_var.columns
vif["vif"] = [
    variance_inflation_factor(
        feature_var.astype(float).values,
        i
    ) for i in range(feature_var.shape[1])
]

# Print and save VIF
print(f"VIF: {vif}")
with open("vif.txt", "w") as f:
    f.write(str(vif))
