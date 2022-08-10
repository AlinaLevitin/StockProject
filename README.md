# StockProject

This project is in collaboration with my father Semyon Goldstein (S.G), a retired software engineer.

I designed a deep-learning code based on stockmarket values that S.G collected.
**I cannot disclose the data collection and manipulation.**
The aim of this project is to asses if deep learning can be employed for the needs of S.G.
This model was not used as as.

**In short:**
The deep learning model I created is written in python and is based on TenserFlow (TF) deep learning library.
I built a small custom package employing TF, pandas and Sklearn in order to train the model to predict potential finantial benefits.
After optimizing the hyperparameters: learning rate, number of epochs, number of neuorons and batch-size, I prooved to S.G that deep learning can be useful for his needs.

I decided that creating a custom package will be the most visually pleasing method.

The stockmarket data that S.G collected was further maniulated to collect "winning" and "losing" instances for training the model.

After some optimizations using a small representative data set, I decided to use three dense layers using "tf.keras.models.Sequential".
The data manipulation done by S.G, generated stockmarket values ranging from roughly -10 to +10 as a function of time.
First I scaled the data to range between -1 and 1.
Standardizing the data to be all positive was also an option, but since some downstream mathematical operations would get difficult to debug, I decided to only scale the numbers.
Since a good portion of the input data is negative, I wanted the first layer to consider negative numbers evenly, therefore the first activation function is "tanh".
Other options included "leaky relu", and "elu". However, I didn't want the model to have negative bias on the negative numbers since at this point negatve numbers are just inputs.
Using an activation function as "Maxout" or "relu" would "kill" neurons with nagative values.
The next two layers were decided by trial and error using a small representative data set.

I decided to use "Adam" gradient descent since I wanted the best optimizer using a dynamic learning rate and momentum during the gradient descent.

Next, the model was optimized by "for looping" the hyperparameters: learning rate, number of epochs, number of neuorons in the deep learning layers and batch-size for stochastic gradient descent.

Finally, the best hyperparameters were used to generate the final model
