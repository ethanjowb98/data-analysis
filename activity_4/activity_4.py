import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plot
from sklearn import metrics

headers = [
    "os_type",
    "data_bytes",
    "time"
]

dataset = pd.read_csv("act4.csv", header=None, names=headers)
dataset.drop(index=dataset.index[0], axis="index", inplace=True)
dataset["os_type"] = dataset["os_type"].astype("category").cat.codes

feature_var = dataset.drop(columns="time")  # X
dependent_var = dataset["time"]  # Y

model = LinearRegression()
model.fit(feature_var, dependent_var)

dep_var_predict = model.predict(feature_var)

plot.scatter(dependent_var, dep_var_predict)
plot.xlabel("Actual Time")
plot.ylabel("Predicted Time")

score = metrics.r2_score(dependent_var, dep_var_predict)

with open("summary.txt", "w") as file:
    file.write(str(float(score)))

plot.savefig("scatterplot.png")
