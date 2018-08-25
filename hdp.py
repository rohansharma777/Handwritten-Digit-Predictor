from sklearn.neural_network import MLPClassifier

import pandas as pd

X = pd.read_csv("Dataset.csv", sep = ",", header = None)

y = pd.read_csv("y.csv", sep= ",", header = None)

model = MLPClassifier(activation = 'logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(25,), random_state=1)

model.fit(X, y)

test = pd.read_csv("testset.csv")

x = model.predict(test)

print(x);
