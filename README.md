# perceptron
To gain experience with Python and neural net learning algorithms without using any libraries.

Create a Python implementation of a Perceptron together with an implementation of a couple Perceptron training algorithms. I will conduct some experiments to see if they confirm or deny what I've learned about the Perceptron's ability to PAC-learn various functions.

python perceptron.py activation training_alg ground_file distribution num_train num_test epsilon

Below is a concrete example of filling in the parameters in this line:

python perceptron.py relu perceptron my_nested_bool_fn.txt bool 500 250 0.2

activation = threshold, tanh, relu
training_alg = perceptron, winnow
distribution = bool, sphere

Experiment 1
design and carry out experiments to see how well the Perceptron update rule lets you learn a nested boolean formula or a linear threshold function under the uniform boolean or uniform unit spherical distributions. For spherical distributions only test with respect to linear threshold function learning. In the perceptron update rule, uniform boolean distribution case in particular, do the experiments jive with the PAC learning results? 

Experiment 2
design and carry out experiments to see how well the Winnow rule learns a linear threshold function under the uniform unit spherical distribution under different choice of activation function.
