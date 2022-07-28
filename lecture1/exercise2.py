import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5))

model.predict([1, 2, 3, 4, 5])
o = model.predict([1, 2, 3, 4, 5])

o = np.array([[1], [2], [3], [4], [5]])
model.predict([1, 2, 3, 4, 5]) / o
model.summary()
model.layers
model.layers[0].weights
