import pandas as pd
import numpy as np
import helper
from sklearn import tree

train = pd.read_csv("../titanic/train.csv")
test = pd.read_csv("../titanic/test.csv")

# train["Age"] = train["Age"].fillna(train["Age"].median())
# train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train['Sex'] == 'male', 'Sex'] = 0
train.loc[train['Sex'] == 'female', 'Sex'] = 1
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2

test["Age"] = test["Age"].fillna(test["Age"].median())
test.loc[test["Sex"] == "male", 'Sex'] = 0
test.loc[test["Sex"] == "female", 'Sex'] = 1
test.loc[test["Embarked"] == "S", 'Embarked'] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2
test.at[152, 'Fare'] = test.Fare.median()

# test = test.dropna()
train = train.dropna()

# helper.deficiency_table(test)

target = train["Survived"].values
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
# 過学習対策
max_depth = 1
min_samples_split = 5

my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

# testから「その2」で使う項目の値を取り出す
test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
# 「その2」の決定木を使って予測をしてCSVへ書き出す
my_prediction_tree_two = my_tree_two.predict(test_features_2)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns = ["Survived"])
my_solution_tree_two.to_csv("my_tree.csv", index_label = ["PassengerId"])
