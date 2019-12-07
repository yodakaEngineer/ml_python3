import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

# 0以上1未満の要素を５個持つ250行のデータを作成
data = np.random.rand(250,5)

# 2.5より大きければ1,小さければ0が入ったLabelを作成
labels = np_utils.to_categorical((np.sum(data, axis=1) > 2.5) * 1)

# model作成
model = Sequential([Dense(20, input_dim=5), Activation('relu'), Dense(2, activation='softmax')])
model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
# nb_epochは回数,validation_splitは検証に使うデータ数
model.fit(data, labels, nb_epoch=300, validation_split=0.2)

test = np.random.rand(200, 5)

predicted_model = model.predict(test)
print(predicted_model)

predict = np.argmax(predicted_model, axis=1)
real = (np.sum(test, axis=1) > 2.5) * 1
# 正解率
print(sum(predict == real) / 200.0)
