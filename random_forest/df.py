import joblib
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



train = pd.read_csv("../titanic/train.csv")
test = pd.read_csv("../titanic/test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

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

train_values = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
target = train["Survived"].values

(X_train, X_test, y_train, y_test) = train_test_split(train_values, target, test_size=0.5, random_state=0)


liparameter = {
    "max_depth": [1,5, None],
    "n_estimators":[1,10,50,100,500],
    # "n_estimators":[1,10,50,100,200,300,400,500],
    # "min_samples_split": [2, 3, 10],
    # "min_samples_split": [1, 5, 10],
    # "min_samples_leaf": [1, 3, 10],
    "min_samples_leaf": [1, 5, 10],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}

licv = GridSearchCV(RandomForestClassifier(), liparameter, cv=5, n_jobs=-1, iid=False)
# licv = RandomForestClassifier(n_estimators = 10,max_depth=1,random_state = 0)

# 0.8226143457042333
# {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 1, 'n_estimators': 50}

licv.fit(X_train, y_train)
print(licv.best_score_)
print(licv.best_params_)
predictor = licv.best_estimator_
result = predictor.predict(X_test)
print(accuracy_score(result,y_test))

joblib.dump(predictor,"predictor_dt_2.pkl",compress=True)
