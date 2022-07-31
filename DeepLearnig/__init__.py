import tensorflow as tf


class DLModel:

    def __init__(self, data, neurons, epochs, learning_rate, batch_size):
        self.data = data
        self.neurons = neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train_and_test(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.neurons, activation='elu', input_shape=(self.data.x_train.shape[1],)),
            tf.keras.layers.Dense(self.neurons, activation='tanh'),
            tf.keras.layers.Dense(self.neurons, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.data.x_train, self.data.y_train,
                  epochs=self.epochs,
                  validation_data=(self.data.x_val, self.data.y_val),
                  batch_size=self.batch_size,
                  verbose=0
                  )
        result = model.evaluate(self.data.x_test, self.data.y_test)

        return result
