# MNIST with tf.Estimator and tf.keras.layers
MNIST tutorial using high level Tensorflow libraries: tf.Estimator and tf.keras.layers

## why those libraries?
these libraries are, in my opinion, the libraries in tensorflow that are flexible enough to be used for most projects while being easy enough to understand, and having simple interfaces

## how to run
`python mnist.py` and that's it!

## what's going on?
`dataset.py` 
 - `download(...)` downloads the MNIST dataset
 - `dataset(...)` checks the data is correct, converts the image from `int` pixel values (0 - 255) to a `float` (0.0 - 1.0), and turns the data into a `tf.data.Dataset`

`mnist.py` 
- `train_data(...)` and `eval_data(...)` get the right data and split it into batches
- `lenet(...)` defines our convolutional neural network model, [lenet](http://yann.lecun.com/exdb/lenet/), as a series of layers
- `model_function(...)` tells our model what to do for training data (learn on the data, output the loss) and eval data (don't learn, just run our model and output our accuracy)
- `main(...)` puts everything together and goes through the data, iteratively training on a batch of training data then testing on a batch of eval data
