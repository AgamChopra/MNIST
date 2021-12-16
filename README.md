# MNIST

Project dependencies: PyTorch, Sklearn, Numpy, PIL, Matplotlib, and gzip.

* my_dataset.py: Contains functions to load MNIST dataset into numpy format, convert custom digits into MNIST format, and function to visualize dataset. Download complete MNIST dataset at http://yann.lecun.com/exdb/mnist/.

* models.py: Cantains various custom ML models and helper functions.

* main.py: Executable file for training the models on the MNIST dataset.

* my_numbers: File containing my handdrawn digits [0,9].


Observed Accuracy:

![stats](https://github.com/AgamChopra/MNIST/blob/main/misc_imgs/stats.PNG?raw=true)


Samploe loss graphs:

Perceptron:

![stats](https://github.com/AgamChopra/MNIST/blob/main/misc_imgs/NN_1_neuron.png?raw=true)


10 layer DenseNN:

![stats](https://github.com/AgamChopra/MNIST/blob/main/misc_imgs/NN_10_layers_500_neurons_per_hiddenlayer.png?raw=true)


10 layer Fully Convolutional NN:

![stats](https://github.com/AgamChopra/MNIST/blob/main/misc_imgs/FCNN_10_layers.png?raw=true)

My numbers:

![stats](https://github.com/AgamChopra/MNIST/blob/main/misc_imgs/number_table.png?raw=true)

Future Work:

Working on implementing a Transformer model [A. Vaswani et al., 2017]


## License

**[The MIT License*](https://github.com/AgamChopra/MNIST/blob/master/LICENSE.md)**
