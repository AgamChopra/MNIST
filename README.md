<h1>MNIST</h1>
<p>Project dependencies: PyTorch, Sklearn, Numpy, PIL, Matplotlib, and gzip.</p>
<ul>
    <li><a href="https://github.com/AgamChopra/MNIST/blob/main/my_dataset.py">my_dataset.py</a>: Contains functions to load MNIST dataset into numpy format, convert custom digits into MNIST
        format, and function to visualize dataset.
        Download complete MNIST dataset at <a href="http://yann.lecun.com/exdb/mnist/" target="blank">here</a>.
    <li><a href="https://github.com/AgamChopra/MNIST/blob/main/models.py">models.py</a>: Cantains various custom ML models and helper functions.
    <li><a href="https://github.com/AgamChopra/MNIST/blob/main/main.py">main.py</a>: Executable file for training the models on the MNIST dataset.
    <li><a href="https://github.com/AgamChopra/MNIST/tree/main/my_numbers">my_numbers</a>: File containing my handdrawn digits [0,9].
</ul>
<p align="center">
    <img width="550" height="80" src="https://github.com/AgamChopra/MNIST/blob/main/misc_imgs/stats.PNG?raw=true">
    <br><i>Fig 1. Observed Accuracy.</i><br><br>
    <img width="180" height="100"
        src="https://github.com/AgamChopra/MNIST/blob/main/misc_imgs/NN_1_neuron.png?raw=true">
    <img width="180" height="100"
        src="https://github.com/AgamChopra/MNIST/blob/main/misc_imgs/NN_10_layers_500_neurons_per_hiddenlayer.png?raw=true">
    <img width="180" height="100"
        src="https://github.com/AgamChopra/MNIST/blob/main/misc_imgs/FCNN_10_layers.png?raw=true">
    <br><i>Fig 2. Loss Plots for Perceptron, 10 layer DenseNN, and 10 layer Fully Convolutional NN, in that
        order.</i><br>
</p>

<h2>Future Work</h2>
<p>Working on implementing a Transformer model <a href="https://arxiv.org/pdf/1706.03762.pdf" target="blank">[A. Vaswani
        et al., 2017]</a></p>
<h2>Lisence</h2>
<p><a href="https://raw.githubusercontent.com/AgamChopra/MNIST/main/LICENSE?token=AFTUZ6KEH5IE4L4ZIZCCUF3BYTS6C" target="blank">[The MIT License]</a></p>
