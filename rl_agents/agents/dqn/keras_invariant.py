"""
Permutation Invariant Neural Network Layer
================================================================================

Core functions and classes.
"""

import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Conv2D, Lambda
from keras.models import Sequential
import numpy as np
import tensorflow as tf


def PermutationInvariant(input_shape, layer_sizes, tuple_dim = 2, reduce_fun = "mean"):
    """
    Implements a permutation invariant layer.
    Each batch in our data consists of `input_shape[0]` observations
    each with `input_shape[1]` features.

    Args:
    input_shape -- A pair of `int` - (number of observations in one batch x
        number of features of each observation). The batch dimension is not included.
    layer_sizes -- A `list` of `int`. Sizes of layers in the dense neural network applied
        to each tuple of observations.
    tuple_dim -- A `int`, how many observations to put in one tuple.
    reduce_fun -- A `string`, type of function to "average" over all tuples.

    Returns:
    g -- A `Sequential` keras container - the permutation invariant layer.
        It consists of one tupple layer that creates all possible `tuple_dim`-tuples
        of observations. On each tuple is applied the same dense neural network
        and then some symmetric pooling function is applied across all of them
        (for example mean or maximum).
    """
    g = Sequential()

    ## Tuple layer
    g.add(Tuples(tuple_dim, input_shape = input_shape))  ## out shape: batch_size x rows x cols

    ## Dense neural net -- implemented with conv layers
    g.add(Lambda(lambda x : K.expand_dims(x, axis = 2))) ## out shape: batch_size x rows x 1 x cols
    for layer_size in layer_sizes:
        g.add(Conv2D(filters = layer_size, kernel_size = (1,1), data_format = "channels_last")) ## out_shape:  batch_size x rows x 1 x layer_size
    g.add(Lambda(lambda x : K.squeeze(x, axis = 2))) ## lout_shape: batch_size x rows x cols

    ## Pooling layer
    if reduce_fun == "mean":
        lambda_layer = Lambda(lambda x : K.mean(x, axis = 1))
    elif reduce_fun == "max":
        lambda_layer = Lambda(lambda x : K.max(x, axis = 1))
    else:
        raise ValueError("Invalid value for argument `reduce_fun` provided. ")
    g.add(lambda_layer) ## out shape: batch_size x cols

    return g


class Tuples(Layer):
    """
    Creates all possible tuples of rows of 2D tensor.

    In the case of tuple_dim = 2, from one input batch:

        x_1,
        x_2,
        ...
        x_n,

    where x_i are rows of the tensor, it creates 2D output tensor:

        x_1 | x_1
        x_1 | x_2
        ...
        x_1 | x_n
        x_2 | x_1
        x_2 | x_2
        ...
        x_2 | x_n
        ...
        x_n | x_n

    Args:
    tuple_dim -- A `int`. Dimension of one tuple (i.e. how many rows from the input
    tensor to combine to create a row in output tensor)
    input_shape -- A `tuple` of `int`. In the most frequent case where our data
        has shape (batch_size x num_rows x num_cols) this should be (num_rows x num_cols).
    """

    def __init__(self, tuple_dim = 2, **kwargs):
        self.tuple_dim = tuple_dim
        super(Tuples, self).__init__(**kwargs)

    def create_indices(self, n, k = 2):
        """
        Creates all integer valued coordinate k-tuples in k dimensional hypercube with edge size n.
        for example n = 4, k = 2
        returns [[0, 0], [0, 1], [0, 2], [0, 3],
                 [1, 0], [1, 1], [1, 2], [1, 3],
                 ...
                 [3, 0], [3, 1], [3, 2], [3, 3]]

        Args:
        n -- A `int`, edge size of the hypercube.
        k -- A `int`, dimension of the hypercube.

        Returns:
        indices_n_k -- A `list` of `list` of `int`. Each inner list represents coordinates of one integer point
            in the hypercube.
        """
        if k == 0:
            indices_n_k = [[]]
        else:
            indices_n_k_minus_1 = self.create_indices(n, k-1)
            indices_n_k = [[i] + indices_n_k_minus_1[c] for i in range(n) for c in range(n**(k-1))]

        return indices_n_k

    def build(self, input_shape):
        # Create indexing tuple
        self.gathering_indices = self.create_indices(input_shape[-2], self.tuple_dim)
        super(Tuples, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, x):
        stack_of_tuples = K.map_fn(
            fn = lambda z :
                K.concatenate(
                    [K.reshape(
                        K.gather(z, i),
                        shape = (1,-1)
                     )
                     for i in self.gathering_indices
                    ],
                    axis = 0
                ),
            elems = x
        )
        return stack_of_tuples


    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1] * self.tuple_dim
        output_shape[-2] = output_shape[-2] ** self.tuple_dim
        return tuple(output_shape)


if __name__ == "__main__":
    tuple_dim = 2
    inp_shape = (2,5,5)

    ############################################################################
    ## TESTING Tuples LAYER

    ## Create graph
    ## Input data, 5 experiments in batch, 3 observations in each experiment, 2 feature columns
    data = tf.placeholder(shape = inp_shape, dtype = tf.float32)
    tuples_layer = Tuples(tuple_dim = tuple_dim, input_shape = inp_shape[1:])
    data_tup = tuples_layer(data)

    ## Run for concrete example
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    np.random.seed(0)
    feed = {data : np.random.randn(*inp_shape)}
    data_eval, data_tup_eval = sess.run([data, data_tup], feed)

    print("-------------------------------------------------------------------")
    print("A) Testing Tuples Layer:")
    print("Tuple dimension is: ", tuple_dim)
    print("Data shape is -- (batch size x num rows x num cols): ", data.shape)
    print("Transformed data shape is -- (batch size x num rows ** tuple size, num cols * tuple size): ", data_tup.shape)
    print("Data value is: ", data_eval)
    print("Transformed data value is:", data_tup_eval)
    ##
    ############################################################################


    ############################################################################
    ## TESTING PermutationInvariant
    layer_sizes = [5, 9, 8, 6]
    perm_inv = PermutationInvariant(input_shape = inp_shape[1:],
                                    layer_sizes = layer_sizes,
                                    tuple_dim = tuple_dim,
                                    reduce_fun = "mean")

    data_perm_inv = perm_inv(data)

    print("-------------------------------------------------------------------")
    print("B) Testing permutation_invariant function:")
    print("Data is the same as in A).")
    print("Shape of permutation invariant layer output: ", data_perm_inv.shape)
    ##
    ############################################################################
