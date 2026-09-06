import tensorflow as tf
import keras

class QNetwork(tf.keras.Model):
    def __init__(self, alpha):
        super(QNetwork, self).__init__()
        self.loss = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=alpha)
    def build(self, input_shape):
        batch_size, state_size = input_shape
        self.f = keras.layers.Flatten(input_shape=(batch_size, state_size))
        self.d1 = keras.layers.Dense(64, activation='tanh', kernel_initializer='he_normal')
        self.d2 = keras.layers.Dense(32, activation='tanh', kernel_initializer='he_normal')
        self.d3 = keras.layers.Dense(3, activation='linear', kernel_initializer='he_normal')  # 3つの行動に対応するQ値を出力
    def call(self, x):
        x = self.f(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
