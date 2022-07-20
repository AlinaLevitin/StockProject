import os
import methods
import tensorflow as tf
import DeepLearnig
import config
import pandas as pd

cwd = os.getcwd()

csv = methods.open_csv(cwd, 'data.csv')

data = DeepLearnig.TrainingData(csv)

acc = []
for i in range(20):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(config.neurons, activation='elu', input_shape=(data.x_train.shape[1],)),
        tf.keras.layers.Dense(config.neurons, activation='tanh'),
        tf.keras.layers.Dense(config.neurons, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(data.x_train, data.y_train,
              epochs=config.epochs,
              validation_data=(data.x_val, data.y_val),
              batch_size=config.batch_size
              )

    result = model.evaluate(data.x_test, data.y_test)
    acc.append(result[1])

df = pd.DataFrame(acc)
DeepLearnig.save_to_csv('acc.csv', df, cwd)

print('###----------------')
print(data)
print('###----------------')
