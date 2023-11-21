"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from libs import Solver






class Linear(object):
    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for a linear (fully-connected) layer.
        Inputs:
        - x: Input to the linear layer, shape (N, d_1, ..., d_k)
        - w: Weights for the linear layer, shape (D, M)
        - b: Biases for the linear layer, shape (M,)
        Returns a tuple of:
        - out: Output from the linear layer, shape (N, M)
        - cache: Object to give to the backward pass (x, w, b)
        """
        out = None
        ######################################################################
        # Linear forward pass
        ######################################################################
        N = x.shape[0]
        x_reshaped = x.view(N, -1)  # Reshape input into rows

        out = torch.mm(x_reshaped, w) + b.unsqueeze(0)  # Forward pass

        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ##################################################
        # Linear backward pass
        ##################################################
        x_reshaped = x.view(x.shape[0], -1)  # Reshape input into rows

        # Gradient of loss w.r.t. x
        dx = torch.mm(dout, w.t())
        dx = dx.view(*x.shape)  # Reshape dx back to input shape

        # Gradient of loss w.r.t. w
        dw = torch.mm(x_reshaped.t(), dout)

        # Gradient of loss w.r.t. b (sum along axis 0)
        db = torch.sum(dout, dim=0)

        ##################################################
        #                END OF YOUR CODE                #
        ##################################################
        return dx, dw, db


class ReLU(object):
    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        ###################################################
        # ReLU forward pass
        ###################################################
        out = torch.max(torch.tensor(0.0, device=x.device), x)  # ReLU activation function
        ###################################################
        #                 END OF YOUR CODE                #
        ###################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        #####################################################
        # ReLU backward pass
        #####################################################
        dx = dout * (x > 0).float()  # Gradient of ReLU function
        #####################################################
        #                  END OF YOUR CODE                 #
        #####################################################
        return dx



class Linear_ReLU(object):
    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs a linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass (hint: cache = (fc_cache, relu_cache))
        """
        out = None
        cache = None
        ######################################################################
        # Linear-ReLU forward pass
        ######################################################################
        fc_out = torch.matmul(x.view(x.shape[0], -1), w) + b
        out = torch.relu(fc_out)
        cache = (x, w, b, fc_out)
        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        dx, dw, db = None, None, None
        ######################################################################
        # Linear-ReLU backward pass
        ######################################################################
        x, w, b, fc_out = cache

        drelu = dout * (fc_out > 0).float()

        dx = torch.matmul(drelu, w.t())
        dx = dx.view(x.shape)
        dw = torch.matmul(x.view(x.shape[0], -1).t(), drelu)
        db = torch.sum(drelu, dim=0)

        ######################################################################
        #                END OF YOUR CODE                                    #
        ######################################################################
        return dx, dw, db





def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss = None
    dx = None
    ######################################################################
    # Softmax layer implementation
    ######################################################################
    N = x.shape[0]

    # Shift values for numerical stability
    shifted_logits = x - torch.max(x, dim=1, keepdim=True)[0]

    # Calculate softmax probabilities
    exp_scores = torch.exp(shifted_logits)
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

    # Compute the loss
    correct_class_probs = probs[range(N), y]
    loss = torch.mean(-torch.log(correct_class_probs))

    # Compute gradient of the loss with respect to x
    dx = probs.clone()
    dx[range(N), y] -= 1
    dx /= N

    ######################################################################
    #                    END OF YOUR CODE                                #
    ######################################################################
    return loss, dx


import torch

class TwoLayerNet(object):
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        self.params = {}
        self.reg = reg

        # Initialize weights and biases
        self.params['W1'] = weight_scale * torch.randn(input_dim, hidden_dim, dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W2'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        scores = None
        # Forward pass
        h1 = torch.relu(torch.matmul(X.view(X.shape[0], -1), self.params['W1']) + self.params['b1'])
        scores = torch.matmul(h1, self.params['W2']) + self.params['b2']
        scores = scores.to(torch.float64)  # Force scores to be of type float64

        if y is None:
            return scores

        loss, grads = 0, {}
        # Compute softmax and cross-entropy loss
        scores -= torch.max(scores, dim=1, keepdim=True)[0]  # Shift scores for numerical stability
        softmax = torch.exp(scores) / torch.exp(scores).sum(dim=1, keepdim=True)
        loss = torch.mean(-torch.log(softmax[torch.arange(X.shape[0]), y]))

        # Backward pass
        dscores = softmax.clone()
        dscores[torch.arange(X.shape[0]), y] -= 1
        dscores /= X.shape[0]

        grads['W2'] = torch.matmul(h1.t(), dscores)
        grads['b2'] = torch.sum(dscores, dim=0)
        dhidden = torch.matmul(dscores, self.params['W2'].t())
        dhidden[h1 <= 0] = 0  # ReLU derivative

        grads['W1'] = torch.matmul(X.view(X.shape[0], -1).t(), dhidden)
        grads['b1'] = torch.sum(dhidden, dim=0)

        # Add regularization to the loss and gradients
        loss += 0.5 * self.reg * (torch.sum(self.params['W1']**2) + torch.sum(self.params['W2']**2))
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']

        return loss, grads




import torch
import torch.nn.functional as F

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        ... (same as your provided code)

        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize the parameters of the network
        dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            self.params[f'W{i + 1}'] = weight_scale * torch.randn(dims[i], dims[i + 1], dtype=dtype, device=device)
            self.params[f'b{i + 1}'] = torch.zeros(dims[i + 1], dtype=dtype, device=device)

        # Dropout
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    # save() and load() methods remain the same as your provided code.
    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'use_dropout': self.use_dropout,
            'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))
    def loss(self, X, y=None):
        scores = None
        # Forward pass
        hidden_layers = [X]
        for i in range(1, self.num_layers):
            hidden = torch.relu(
                torch.matmul(hidden_layers[i - 1], self.params[f'W{i}']) + self.params[f'b{i}']
            )
            hidden_layers.append(hidden)

        scores = torch.matmul(hidden_layers[-1], self.params[f'W{self.num_layers}']) + self.params[f'b{self.num_layers}']
        scores = scores.to(torch.float64)  # Force scores to be of type float64

        if y is None:
            return scores

        loss, grads = 0, {}
        # Compute softmax and cross-entropy loss
        scores -= torch.max(scores, dim=1, keepdim=True)[0]  # Shift scores for numerical stability
        softmax = torch.exp(scores) / torch.exp(scores).sum(dim=1, keepdim=True)
        loss = torch.mean(-torch.log(softmax[torch.arange(X.shape[0]), y]))

        # Backward pass
        dscores = softmax.clone()
        dscores[torch.arange(X.shape[0]), y] -= 1
        dscores /= X.shape[0]

        for i in range(self.num_layers, 0, -1):
            grads[f'W{i}'] = torch.matmul(hidden_layers[i - 1].t(), dscores)
            grads[f'b{i}'] = torch.sum(dscores, dim=0)
            dhidden = torch.matmul(dscores, self.params[f'W{i}'].t())
            dhidden[hidden_layers[i - 1] <= 0] = 0  # ReLU derivative
            dscores = dhidden

        # Add regularization to the loss and gradients
        reg_loss = 0.5 * self.reg * sum(torch.sum(self.params[f'W{i}']**2) for i in range(1, self.num_layers + 1))
        loss += reg_loss

        for i in range(1, self.num_layers + 1):
            grads[f'W{i}'] += self.reg * self.params[f'W{i}']

        return loss, grads
def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)

    # Define the Solver parameters
    solver_params = {
        'update_rule': sgd,
        'optim_config': {'learning_rate': 1e-1},
        'lr_decay': 0.99,
        'num_epochs': 5,
        'batch_size': 100,
        'print_every': 1000,
        'device': device
    }

    # Create the Solver instance
    solver = Solver(model, data_dict,
                    update_rule=solver_params['update_rule'],
                    optim_config=solver_params['optim_config'],
                    lr_decay=solver_params['lr_decay'],
                    num_epochs=solver_params['num_epochs'],
                    batch_size=solver_params['batch_size'],
                    print_every=solver_params['print_every'],
                    device=solver_params['device'])

    # Train the model
    solver.train()
    return solver



def get_three_layer_network_params():
    weight_scale = 1e-3  # Reduce weight scale for overfitting
    learning_rate = 1e-1  # Increase learning rate for faster learning

    return weight_scale, learning_rate



def get_five_layer_network_params():
    ################################################################
    # TODO: Change weight_scale and learning_rate so your          #
    # model achieves 100% training accuracy within 20 epochs.      #
    ################################################################
    learning_rate = 2e-3  # Experiment with this!
    weight_scale = 1e-5   # Experiment with this!
    ################################################################
    #                       END OF YOUR CODE                       #
    ################################################################
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A torch array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    next_w = None
    ##################################################################
    # TODO: Implement the momentum update formula. Store the         #
    # updated value in the next_w variable. You should also use and  #
    # update the velocity v.                                         #
    ##################################################################
    # Momentum update
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    v = momentum * v - learning_rate * dw  # Update velocity
    next_w = w + v  # Update weights
    ###################################################################
    #                           END OF YOUR CODE                      #
    ###################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    next_w = None

    # RMSProp update formula
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw**2
    next_w = w - config['learning_rate'] * dw / (torch.sqrt(config['cache']) + config['epsilon'])

    return next_w, config



def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None

    # Adam update formula
    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dw**2)
    m_hat = config['m'] / (1 - config['beta1']**config['t'])
    v_hat = config['v'] / (1 - config['beta2']**config['t'])
    next_w = w - config['learning_rate'] * m_hat / (torch.sqrt(v_hat) + config['epsilon'])

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
            if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            mask = (torch.rand_like(x) > p) / (1 - p)

            out = x * mask
        elif mode == 'test':
            out = x

        cache = (dropout_param, mask)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            dx = dout * mask
        elif mode == 'test':
            dx = dout

        return dx

