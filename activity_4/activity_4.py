import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plot
from sklearn import metrics
from statsmodels import api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

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
plot.clf()

# Calculate R-squared and save to summary.txt
r2_score = metrics.r2_score(dep_var, dep_predict)

# Get coefficients and intercept
coeff = model.coef_
intercept = model.intercept_

# Convert dependent variable to float
dep_var = dep_var.astype(float)

# Calculate residuals
residuals = dep_var - dep_predict

# Kolmogorov-Smirnov Test
ks_stats, ks_pvalue = stats.kstest(residuals, "norm")

# Shapiro-Wilk Test
sw_stats, sw_pvalue = stats.shapiro(residuals)

# Plot Q-Q plot of residuals
sm.qqplot(residuals, line="45")
plot.title("Q-Q Plot of Residuals")
plot.savefig("qqplot_residuals.png")
plot.clf()

# Plot residuals vs. fitted values to check homoscedasticity
plot.scatter(dep_predict, residuals)
plot.xlabel("Fitted values")
plot.ylabel("Residuals")
plot.title("Residuals vs Fitted Values")
plot.savefig("homoscedasticity.png")
plot.clf()

# Calculate Variance Inflation Factor (VIF) to check multicollinearity
vif = pd.DataFrame()
vif["feature"] = feature_var.columns
vif["vif"] = [
    variance_inflation_factor(
        feature_var.astype(float).values,
        i
    ) for i in range(feature_var.shape[1])
]

# Write it to file
with open("summary.txt", "w") as file:
    fw = file.write
    fw("SUMMARY\n")
    fw(f"R2_Score: \t\t\t\t\t{r2_score}\n")
    fw("-------------\n")
    fw("Coeff: \t\t\t[\n")
    for i, value in enumerate(coeff):
        fw(f"\t{headers[i]}: {value},\n ")
    fw("]\n")
    fw(f"Intercept: \t\t\t\t\t{intercept}\n")
    fw("-------------\n")
    fw("VIF\n")
    fw(f"{vif}\n")
    fw("-------------\n")
    fw("Kolmogorov-Smirnov Test\n")
    fw(f"Value: \t\t\t\t\t\t{ks_pvalue}\n")
    fw(f"Interpretation: \t\t\t{'Normally distirbuted' if ks_pvalue > 0.05 else 'Not normally distributed'}\n")
    fw("-------------\n")
    fw("Shapiro-Wil Test\n")
    fw(f"Value: \t\t\t\t\t\t{sw_pvalue}\n")
    fw(f"Interpretation: \t\t\t{'Normally distirbuted' if sw_pvalue > 0.05 else 'Not normally distributed'}\n")
