import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=10, activation='relu'),
    Dense(units=3, activation='linear')
])

X = np.array([
    [20, 15],
    [20, 17],
    [10, 10],
    [40, 30],
    [50, 2]
    ])

Y = np.array([0, 1, 1, 2, 0])

model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(X,Y,epochs=100)
logits=model(X)
f_x = tf.nn.softmax(logits)

x_new = np.array([[20, 3]])
print(model.predict(x_new))
