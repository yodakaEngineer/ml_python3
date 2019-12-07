import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

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

data = train_values
labels = np_utils.to_categorical(target)
passengerId = test["PassengerId"]
test = test[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values


model = Sequential([
    Dense(32, input_dim=7),
    Activation('relu'),
    Dropout(0.2),
    Dense(16),
    Activation('relu'),
    Dense(2, activation='softmax')
])
model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=300, validation_split=0.2)

predicted_model = model.predict(test)

predict = np.argmax(predicted_model, axis=1)
PassengerId = np.array(passengerId).astype(int)
my_solution_tree_two = pd.DataFrame(predict, PassengerId, columns = ["Survived"])
my_solution_tree_two.to_csv("dl.csv", index_label = ["PassengerId"])
