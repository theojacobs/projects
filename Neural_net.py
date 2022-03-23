import pdb
import time
from tqdm.notebook import tqdm
import numpy as np

DATA_TYPE = np.float32
EPSILON = 1e-12


def xavier(shape, seed=None):
    n_in, n_out = shape
    if seed is not None:
        # set seed to fixed number (e.g. layer idx) for predictable results
        np.random.seed(seed)
    l_bound, u_bound = -np.sqrt(6.0/(n_in+n_out)), (np.sqrt(6.0/(n_in+n_out)))
    # todo initialize uniformly at random from [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
    return l_bound, u_bound


# InputValue: These are input values. They are leaves in the computational graph.
#              Hence we never compute the gradient wrt them.
class InputValue:
    def __init__(self, value=None):
        self.value = DATA_TYPE(value).copy()
        self.grad = None

    def set(self, value):
        self.value = DATA_TYPE(value).copy()


# Parameters: Class for weight and biases, the trainable parameters whose values need to be updated
class Param:
    def __init__(self, value):
        self.value = DATA_TYPE(value).copy()
        self.grad = DATA_TYPE(0)


'''
  Class name: Add
  Class usage: add two matrices a, b with broadcasting supported by numpy "+" operation.
  Class function:
      forward: calculate a + b with possible broadcasting
      backward: calculate derivative w.r.t to a and b
'''


class Add:  # Add with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value + self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad

        if self.b.grad is not None:
            self.b.grad = self.b.grad + self.grad


'''
Class Name: Mul
Class Usage: elementwise multiplication with two matrix 
Class Functions:
    forward: compute the result a*b
    backward: compute the derivative w.r.t a and b
'''


class Mul:  # Multiply with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value * self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad * self.b.value

        if self.b.grad is not None:
            self.b.grad = self.b.grad + self.grad * self.a.value    
            
            
'''
Class Name: VDot
Class Usage: matrix multiplication where a is a vector and b is a matrix
    b is expected to be a parameter and there is a convention that parameters come last. 
    Typical usage is a is a feature vector with shape (f_dim, ), b a parameter with shape (f_dim, f_dim2).
Class Functions:
     forward: compute the vector matrix multplication result
     backward: compute the derivative w.r.t a and b, where derivative of a and b are both matrices 
'''


class VDot:  # Matrix multiply (fully-connected layer)
    def __init__(self, a, b):
        # a is (m, 1)
        self.a = a
        # b is (m, n)
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None
    
    def forward(self):
        # get the  product of the aW
        z = np.matmul(self.a.value.T, self.b.value)
        self.value = z

    def backward(self):
        if self.a.grad is not None:
            dot = np.dot(self.b.value, self.grad)
            self.a.grad += self.a.grad + dot

        if self.b.grad is not None:
            a_shape = np.reshape(self.a.value, [self.a.value.shape[0], 1])
            b_shape = np.reshape(self.grad, [self.grad.shape[0], 1])
            mat_mul = np.matmul(a_shape, b_shape.T)
            self.b.grad += self.b.grad + mat_mul


'''
Class Name: Sigmoid
Class Usage: compute the elementwise sigmoid activation. Input is vector or matrix. 
    In case of vector, [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = 1/(1 + exp(-a_{i}))
Class Functions:
    forward: compute activation b_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix a  
'''

class Sigmoid:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None


    def forward(self):
        #todo 
        self.value =  1 / (1+np.exp((-1*(self.a.value))))

    def backward(self):

        if self.a.grad is not None:
            # derivative of sigmoid 
            #self.forward()
            sig_prime = 1 / (1+np.exp((-1*(self.a.value))))
            deriv = sig_prime * (1 - sig_prime)
            #deriv = self.value * (1- self.value)
            self.a.grad += self.grad*deriv
        


'''
Class Name: RELU
Class Usage: compute the elementwise RELU activation. Input is vector or matrix. In case of vector, 
    [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = max(0, a_{i})
Class Functions:
    forward: compute activation b_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix a  
'''

class RELU:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        # todo
        self.value = np.maximum(0.0, self.a.value)
        # self.value =

    def backward(self):
        if self.a.grad is not None:
            if self.grad <=  0.0:
                deriv = 0.0
            else:
                deriv = self.grad
                
            self.a.grad += deriv 


'''
Class Name: SoftMax
Class Usage: compute the softmax activation for each element in the matrix, normalization by each all elements 
    in each batch (row). Specifically, input is matrix [a_{00}, a_{01}, ..., a_{0n}, ..., a_{b0}, a_{b1}, ..., a_{bn}], 
    output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(a_{bi})/(exp(a_{b0}) + ... + exp(a_{bn}))
Class Functions:
    forward: compute probability p_{bi} for all b, i.
    backward: compute the derivative w.r.t input matrix a 
'''


class SoftMax:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        #todo
        #this may not be numerically stable
        exps = np.exp(self.a.value)
        sum_exp = np.sum(exps)
        self.value = exps/sum_exp

    def backward(self):
        if self.a.grad is not None:
            deriv = self.value*self.grad - np.sum(self.value*self.grad)*self.value
            self.a.grad += deriv

'''
Class Name: Log
Class Usage: compute the elementwise log(a) given a.
Class Functions:
    forward: compute log(a)
    backward: compute the derivative w.r.t input vector a
'''
class Log: # Elementwise Log
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        # todo
        #raise NotImplementedError
        self.value = np.log(self.a.value)

    def backward(self):
        if self.a.grad is not None:
            # todo
            self.a.grad += self.grad *(1/(self.a.value))



'''
Class Name: Aref
Class Usage: get some specific entry in a matrix. a is the matrix with shape (batch_size, N) and idx is vector containing 
    the entry index and a is differentiable.
Class Functions:
    forward: compute a[batch_size, idx]
    backward: compute the derivative w.r.t input matrix a
'''

class Aref:
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None if a.grad is None else DATA_TYPE(0)

    def forward(self):
        xflat = self.a.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        self.pick = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.a.grad is not None:
            grad = np.zeros_like(self.a.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.a.grad = self.a.grad + grad


'''
Class Name: Accuracy
Class Usage: check the predicted label is correct or not. a is the probability vector where each probability is 
            for each class. idx is ground truth label.
Class Functions:
    forward: find the label that has maximum probability and compare it with the ground truth label.
    backward: None 
'''


class Accuracy:
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None
        self.value = None

    def forward(self):
        self.value = np.mean(np.argmax(self.a.value, axis=-1) == self.idx.value)

    def backward(self):
        pass


# Set of allowed/implemented activation functions
ACTIVATIONS = {'relu': RELU,
               'sigmoid': Sigmoid}


class NN:
    def __init__(self, nodes_array, activation):
        # assert nodes_array is a list of positive integers
        assert all(isinstance(item, int) and item > 0 for item in nodes_array)
        # assert activation is supported
        assert activation in ACTIVATIONS.keys()
        self.nodes_array = nodes_array
        self.activation = activation
        self.activation_func = ACTIVATIONS[self.activation]
        self.layer_number = len(nodes_array) - 1
        self.weights = []
        # list of trainable parameters
        self.params = []
        # list of computational graph
        self.components = []
        self.sample_placeholder = InputValue()
        self.label_placeholder = InputValue()
        self.pred_placeholder = None
        self.loss_placeholder = None
        self.accy_placeholder = None

    # helper function for creating a parameter and add it to the list of trainable parameters
    def nn_param(self, value):
        param = Param(value)
        self.params.append(param)
        return param

    # helper function for creating a unary operation object and add it to the computational graph
    def nn_unary_op(self, op, a):
        unary_op = op(a)
        print(f"Append <{unary_op.__class__.__name__}> to the computational graph")
        self.components.append(unary_op)
        return unary_op

    # helper function for creating a binary operation object and add it to the computational graph
    def nn_binary_op(self, op, a, b):
        binary_op = op(a, b)
        print(f"Append <{binary_op.__class__.__name__}> to the computational graph")
        self.components.append(binary_op)
        return binary_op

    def set_weights(self, weights):
        """
        :param weights: a list of tuples (matrices and vectors)
        :return:
        """
        weights = np.array(weights)
        # assert weights have the right shapes
        if len(weights) != self.layer_number:
            raise ValueError(f"You should provide weights for {self.layer_number} layers instead of {len(weights)}")
        for i, item in enumerate(weights):
            weight, bias = item
            if weight.shape != (self.nodes_array[i], self.nodes_array[i + 1]):
                raise ValueError(f"The weight for the layer {i} should have shape ({self.nodes_array[i]}, {self.nodes_array[i + 1]}) instead of {weight.shape}")
            if bias.shape != (self.nodes_array[i + 1],):
                raise ValueError(f"The bias for the layer {i} should have shape ({self.nodes_array[i + 1]}, ) instead of {bias.shape}")

        # reset params to empty list before setting new values
        self.params = []
        # add Param objects to the list of trainable paramters with specified values
        for item in weights:
            weight, bias = item
            weight = self.nn_param(weight)
            bias = self.nn_param(bias)

    def get_weights(self):
        # Assuming params are np.arrays of weights and biases 
        weights = []
        # Extract weight values from the list of Params
        for i, paramObj in enumerate(self.params):
            if i%2 == 0:
                w_b = (paramObj.value, self.params[i+1].value)
                weights.append(w_b)
        return weights

    def init_weights_with_xavier(self):
        weights = []
        #range is 3 for layers - 1
        for i in range(self.layer_number):
            lower, upper = xavier((self.nodes_array[i], self.nodes_array[i+1]))
            w = np.random.uniform(low = lower, high = upper, size=(self.nodes_array[i], self.nodes_array[i+1])).astype(DATA_TYPE)
            b = np.random.uniform(low = lower, high = upper, size=(self.nodes_array[i+1],)).astype(DATA_TYPE)
            weights.append((w, b))
        self.set_weights(weights)

    def build_computational_graph(self):
        if len(self.params) != self.layer_number*2:
            raise ValueError("Trainable Parameters have not been initialized yet. Call init_weights_with_xavier first.")

        # Reset computational graph to empty list
        self.components = []

        prev_output = self.sample_placeholder
        # You need to call self.nn_binary_op(VDot, x, y), self.nn_binary_op(Add, x, y),
        # self.nn_unary_op(self.activation_func, x), self.nn_unary_op(SoftMax, x) here to construct the neural network
        # You code should support different number of layers
        for i in range(self.layer_number): 
            weight, bias = self.params[i*2], self.params[i*2 + 1]
            vdot = self.nn_binary_op(VDot, prev_output, weight)
            with_bias = self.nn_binary_op(Add, vdot, bias)
            if i != self.layer_number-1:
                prev_output = self.nn_unary_op(self.activation_func, with_bias)
            else:
                prev_output = self.nn_unary_op(SoftMax, with_bias)

        pred = prev_output
        return pred

    def cross_entropy_loss(self):
        # # You need to construct cross entropy loss using self.pred_placeholder and self.label_placeholder
        # # as well as self.nn_binary_op and self.nn_unary_op
        label_prob = self.nn_binary_op(Aref, self.pred_placeholder, self.label_placeholder)
        log_prob = self.nn_unary_op(Log, label_prob)
        loss = self.nn_binary_op(Mul, log_prob, InputValue(-1))
        return loss


    def eval(self, X, y):
        if len(self.components)==0:
            raise ValueError("Computational graph not built yet. Call build_computational_graph first.")
        accuracy = 0.
        objective = 0.
        for k in range(len(y)):
            self.sample_placeholder.set(X[k])
            self.label_placeholder.set(y[k])
            self.forward()
            accuracy += self.accy_placeholder.value
            objective += self.loss_placeholder.value
        accuracy /= len(y)
        objective /= len(y)
        return accuracy, objective

    def fit(self, X, y, alpha, t):
        """
            Use the cross entropy loss.  The stochastic
            gradient descent should go through the examples in order, so
            that your output is deterministic and can be verified.
        :param X: an (m, n)-shaped numpy input matrix
        :param y: an (m,1)-shaped numpy output
        :param alpha: the learning rate
        :param t: the number of iterations
        :return:
        """
        # create sample and input placeholder
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in tqdm(range(X.shape[0])):
                # tqdm adds a progress bar
                for p in self.params:
                    p.grad = DATA_TYPE(0)
                for c in self.components:
                    if c.grad is not None:
                        c.grad = DATA_TYPE(0)    
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])
                self.forward()
                self.backward(self.loss_placeholder)
                self.sgd_update_parameter(alpha)
                # todo make function calls to complete the training process

            # evaluate on train set
            avg_acc, avg_loss = self.eval(X, y)
            print("Epoch %d: train loss = %.4f, accy = %.4f, [%.3f secs]" % (epoch, avg_loss, avg_acc, time.time()-since))
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            since = time.time()

    def forward(self):
        for c in self.components:
            c.forward()

    def backward(self, loss): 
        # setting initial gradient that gets backpropogated all the way through 
        loss.grad = np.ones_like(loss.value)
        for component in self.components[::-1]:
            component.backward()

    # Optimization functions
    def sgd_update_parameter(self, lr):
        for param in self.params:
            param.value -= param.grad * lr

    def gradient_estimate(self, param, epsilon=EPSILON):
        # todo optional, could be used for debugging
        pass


def test_set_and_get_weights():
    # we will change nodes_array in our test
    nodes_array = [4, 5, 5, 3]
    nn = NN(nodes_array, activation="sigmoid")
    # make sure to have the same datatype as DT
    # cotains np array of shape (784, 128)
    weights = []
    #range is 3 for layers - 1
    for i in range(nn.layer_number):
        # makes a 4 x 5 array of weights
        w = np.random.random((nodes_array[i], nodes_array[i+1])).astype(DATA_TYPE)

        # makes a random array of biases 1 x 5 array f
        b = np.random.random((nodes_array[i+1],)).astype(DATA_TYPE)
        # for each layer there is a tuple with the weight
        # and the bias to be multiplied and added
        weights.append((w, b))

    nn.set_weights(weights)
    nn_weights = nn.get_weights()

    for i in range(nn.layer_number):
        weight, bias = weights[i]
        nn_weight, nn_bias = nn_weights[i]
        if not np.array_equal(weight, nn_weight):
            raise AssertionError(f"The weight on layer {i} is not consistent.\n Set as {weight}, returned as {nn_weight}")
        if not np.array_equal(bias, nn_bias):
            raise AssertionError(f"The bias on layer {i} is not consistent.\n Set as {bias}, returned as {nn_bias}")
    print("Passed the test for set_weights and get_weights.")
    print(nn.params)
    print(nn.weights)



def main():
    test_set_and_get_weights()


if __name__ == "__main__":
    main()