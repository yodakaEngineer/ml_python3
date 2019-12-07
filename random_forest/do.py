import joblib
import numpy as np
import pandas as pd

test = pd.read_csv("../titanic/test.csv")
clf = joblib.load("predictor_dt.pkl")

test["Age"] = test["Age"].fillna(test["Age"].median())
test.loc[test["Sex"] == "male", 'Sex'] = 0
test.loc[test["Sex"] == "female", 'Sex'] = 1
test.loc[test["Embarked"] == "S", 'Embarked'] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2
test.at[152, 'Fare'] = test.Fare.median()

test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred = clf.predict(test_features_2)

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(pred, PassengerId, columns = ["Survived"])
my_solution_tree_two.to_csv("random_forest.csv", index_label = ["PassengerId"])