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
        tf.keras.layers.Dense(config.NEURONS, activation='elu', input_shape=(data.x_train.shape[1],)),
        tf.keras.layers.Dense(config.NEURONS, activation='tanh'),
        tf.keras.layers.Dense(config.NEURONS, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(data.x_train, data.y_train,
                        epochs=config.EPOCHS,
                        validation_data=(data.x_val, data.y_val),
                        batch_size=config.BATCH_SIZE,
                        verbose=0
                        )

    result = model.evaluate(data.x_test, data.y_test, verbose=0)
    acc.append(result[1])

accuracy_average = methods.average(acc)

summary = {'average accuracy': accuracy_average,
           'training data': data.train_num,
           'validation data': data.val_num,
           'test data': data.test_num,
           'steps back': config.STEPS_BACK,
           'steps forward': config.STEPS_FORWARD,
           'epochs': config.EPOCHS,
           'neurons': config.NEURONS,
           'learning_rate': config.LEARNING_RATE,
           }

df = pd.DataFrame([summary])
methods.save_to_csv('summary.csv', df, cwd)
print('FINISHED!')
