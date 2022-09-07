# StockProject

This project is in collaboration with my father Semyon Goldstein (S.G), a retired software engineer.

I designed a deep-learning code based on stockmarket values that S.G collected.
**I cannot disclose the data collection and manipulation.**
The aim of this project is to assess if deep learning can be employed for the needs of S.G.

**In short:**
The deep learning model I created is written in python and is based on TenserFlow (TF) deep learning library.
I built a small custom package employing TF, pandas, numpy and Sklearn in order to train the model to predict potential financial benefits.
After optimizing the hyperparameters: learning rate, number of epochs, number of neurons and batch-size, I proved to S.G that deep learning can be useful for his needs.

**In long:** :)
I decided to use OOP and created a custom deep learning package (DeepLearning.py) and neural network handling API (NeuralNetwork_API.py) for S.G to use.
The stockmarket data that S.G collected was further manipulated to collect "winning" and "losing" instances as outputs.
After some optimizations using a small representative data set, I decided to use five dense layers using "tf.keras.models.Sequential".
The data manipulation done by S.G, generated stockmarket values ranging from roughly -10 to +10 as a function of time.
First I scaled the data to range between -1 and 1.
Standardizing the data to be all positive was also an option, but since some downstream mathematical operations would get difficult to debug, I decided to only scale the numbers.
Since a good portion of the input data is in the negative range, I wanted the first layer to consider negative numbers evenly, therefore the first activation function is "tanh".
Other options included "leaky relu", and "elu". However, I didn't want the model to have negative bias on the negative numbers inputs.
Using an activation function as "Maxout" or "relu" would "kill" neurons with negative values.
The next four layers were decided by trial and error.

I decided to use "Adam" gradient descent since I wanted the best optimizer using a dynamic learning rate and momentum during the gradient descent.

Next, the model was optimized by "for looping" and changing the hyperparameters: learning rate, number of epochs, number of neurons in the deep learning layers and batch-size for stochastic gradient descent.

Finally, the best hyperparameters were used to generate the final model and the model was trained on the full set or training data (2m data-points).
The API I created for S.G enables to read training data, train the model and save it, and suggest stockmarket positions based on the trained neaural network.
