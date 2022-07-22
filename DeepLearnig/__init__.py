import tensorflow as tf


class DLModel:

    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, neurons, epochs, learning_rate, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.neurons = neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train_and_test(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.neurons, activation='elu', input_shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(self.neurons, activation='tanh'),
            tf.keras.layers.Dense(self.neurons, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train,
                  epochs=self.epochs,
                  validation_data=(self.x_val, self.y_val),
                  batch_size=self.batch_size,
                  verbose=0
                  )
        result = model.evaluate(self.x_test, self.y_test)

        return result
