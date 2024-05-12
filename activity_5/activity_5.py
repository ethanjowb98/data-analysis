# Code Tutorial: https://www.datacamp.com/tutorial/understanding-logistic-regression-python

import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plot
from sklearn import metrics

headers = [
    "spam",
    "to_multiple",
    "image",
    "attach",
    "password",
    "line_breaks",
    "format",
    "re_subj",
    "urgent_subj"
]

dataset = pd.read_excel("./activity_5_data.xlsx", names=headers, header=None)
dataset.drop(index=dataset.index[0], axis="index", inplace=True)
dataset = dataset.dropna()

for row in ["image", "attach", "password"]:
    dataset.loc[dataset[row] > 1, row] = 1

feature_headers = headers[1:]

ind_var = dataset[feature_headers]  # Feature (X)
dep_var = dataset["spam"]  # Target (Y)

ind_var_train, ind_var_test, dep_var_train, dep_var_test = train_test_split(
    ind_var,
    dep_var,
    test_size=0.25,
    random_state=16
)

logreg = LogisticRegression(random_state=3, max_iter=1000)
logreg.fit(ind_var, dep_var.tolist())

dep_var_predict = logreg.predict(ind_var)

conf_matrix = metrics.confusion_matrix(dep_var.tolist(), dep_var_predict.tolist())

sb.set_theme(style="white")
sb.set_theme(style="whitegrid", color_codes=True)

class_names = [0, 1]
fig, ax = plot.subplots()
tick_marks = np.arange(len(class_names))
plot.xticks(ticks=tick_marks, labels=class_names)
plot.yticks(ticks=tick_marks, labels=class_names)

sb.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt="g")
ax.xaxis.set_label_position("top")

plot.title("Confusion matrix")
plot.ylabel("Actual data")
plot.xlabel("Predicted data")

report = metrics.classification_report(
    y_true=dep_var.tolist(),
    y_pred=dep_var_predict.tolist(),
    target_names=["not spam", "spam"]
)

with open("summary.txt", "w") as file:
    file.write(report)

plot.savefig("heatmap.png")
