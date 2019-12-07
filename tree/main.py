import pandas as pd
import numpy as np
import helper
from sklearn import tree

# trainを読み込んで、testの乗客たちの生存予測を行う

train = pd.read_csv("../titanic/train.csv")
test = pd.read_csv("../titanic/test.csv")

# 最初の５行見る
# print(train.head())

# 統計情報を見る
# print(train.describe())

# helper.deficiency_table(train)
# helper.deficiency_table(test)

# 欠損値補完
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

# helper.deficiency_table(train)

# 文字列データを数字に
train.loc[train['Sex'] == 'male', 'Sex'] = 0
train.loc[train['Sex'] == 'female', 'Sex'] = 1
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2

print(train.head(10))

test["Age"] = test["Age"].fillna(test["Age"].median())
test.loc[test["Sex"] == "male", 'Sex'] = 0
test.loc[test["Sex"] == "female", 'Sex'] = 1
test.loc[test["Embarked"] == "S", 'Embarked'] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2
test.at[152, 'Fare'] = test.Fare.median()

print(test.head(10))

# 決定木モデルを使った予測

# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)

print(my_prediction.shape)
print(my_prediction)

# PassengerIdを取得
# PassengerId = np.array(test["PassengerId"]).astype(int)
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
# my_tree_one.csvとして書き出し
# my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])


features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

# 過学習対策
# max_depth = 1
# min_samples_split = 5
#
# my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
# my_tree_two = my_tree_two.fit(features_two, target)
#
# # testから「その2」で使う項目の値を取り出す
test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
# # 「その2」の決定木を使って予測をしてCSVへ書き出す
# my_prediction_tree_two = my_tree_two.predict(test_features_2)
# PassengerId = np.array(test["PassengerId"]).astype(int)
# my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns = ["Survived"])
# my_solution_tree_two.to_csv("my_tree_three.csv", index_label = ["PassengerId"])

# 1 0.76076
# 3 0.76555

# 実験結果
# 多分max_depthは分類木だから１の方がいい
# 実際結果も微々たるものだけど１の方がよかった。
# min_samples_splitはそれほど関係ない？

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 10,max_depth=1,random_state = 0)
clf = clf.fit(features_two , target)
pred = clf.predict(test_features_2)

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(pred, PassengerId, columns = ["Survived"])
my_solution_tree_two.to_csv("random_forest.csv", index_label = ["PassengerId"])

# fpr, tpr , thresholds = roc_curve(test_y,pred,pos_label = 1)
# auc(fpr,tpr)
# accuracy_score(pred,test_y)
#
# DecisionTreeClassifier