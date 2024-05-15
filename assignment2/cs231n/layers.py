from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    x_shape = x.shape
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ! Linear Combination + bias
    x_flatten = x.reshape(x.shape[0], -1)
    out = x_flatten @ w + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x.reshape(x_shape), w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

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
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #! out = xw + b -> dout = [dout/dx, dout/dw] = [w, x]
    #? dx(N, d_1, ..., d_k) = dout(N, M) @ w.T(M, D)
    dx = (dout @ w.T)
    dx = dx.reshape(x.shape)

    #? dw(D, M) = dout.T(M, N) @ x(N, D) => (M, D)
    dw = (dout.T @ x.reshape((x.shape[0], -1))).T

    #! db = 1
    #? db(M,) = dout * 1(N, M) = dout(N, M) -> row(N) is collapsed to (M,)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # TODO: Does handling the zero to prevent gradient vanishing problem needed? 
    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #! out = +x if x > 0 else 0
    #? 기본적으로 forward pass상에서 branch가 expand되었으면, backward pass상에서는 branch가 collapse된다. 
    #? 따라서 forward pass 상에서 x가 어느 차원으로 확장되거나 축소되었는지를 알면,
    #? backward pass 상에서 x가 어느 차원으로 축소되거나 확장되어 영향을 어떻게 합쳐야 할지를 알 수 있다. 
    #? 여기서의 relu함수는 branch를 늘리지도 줄이지도 않고 있으므로 np.sum과 같이 collapse시킬 필요가 없다.
    x_activated = np.where(x>0, 1, 0)
    dx = dout * x_activated

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****Forwardpass*****
    #! forward-pass initialization
    N, _ = x.shape

    #! forward-pass1 -> x_exp(N, C) 
    #? For numerical stability
    #* number which exploded can cause overflow, underflow or precision loss
    x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))

    #! forward-pass2 -> x_exp_sum(N, 1)
    x_exp_sum = np.sum(x_exp, axis=1, keepdims=True)

    #! forward-pass3 -> softmax function f_j(z) = e^z_j / summation_k(e^z_k)
    #? softmax(N, C) = x_exp(N, C) / x_exp_sum(N, C: broadcasted)
    softmax = x_exp / x_exp_sum

    #! forward-pass4 -> softmax_true_class
    #? softmax_true_class(N, 1)
    softmax_true_class = softmax[range(N), y][:, np.newaxis] # keep dimension

    #! forward-pass5 -> cross_entropy_loss L_i = -f_y + log(summation(e^f_j))
    #? cross_entropy_loss(N, 1) = -softmax[range(N), y](N, 1) + log(x_exp_sum)(N, 1)
    #? cross_entropy_loss(N, 1) = -np.log(softmax[range(N), y])(N, 1)
    cross_entropy_loss = -np.log(softmax_true_class)

    #! forward-pass5
    loss = np.sum(cross_entropy_loss, axis=0) / N

    ###########################################################################
    # # *****Backprop*****

    # softmax[np.arange(N), y] -= 1 # 재민 코드
    # dx = softmax / N # 재민 코드

    #! backprop initialization
    #? dloss(1,)
    dloss = 1. / N

    #! backprop - loss and cross_entropy_loss -> (summation)
    #? dcross_entropy_loss(N, 1) = dloss(1,) * (N, 1)
    dcross_entropy_loss = dloss * np.ones_like(cross_entropy_loss)

    #! backprop - cross_entropy_loss and softmax[range(N), y] -> (cross_entropy_loss = -np.log(softmax[range(N), y]))
    # TODO: 풀어서 쓴 식과 아닌 식의 진행이 다름. - 손으로 풀어서 써보고 아래 주석처리한 원래 코드에서 문제점을 찾을 것.
    #? dsoftmax[range(N), y](N, 1) = -dcross_entropy_loss(N,) * (1/softmax_true_class)(N,)
    dsoftmax = np.zeros_like(softmax)
    dscores_true = -np.squeeze(dcross_entropy_loss) # * (np.squeeze(1./softmax_true_class))
    
    #! backprop - cross_entropy_loss and log(x_exp_sum) -> cross_entropy_loss = ... + LSE
    #? dLSE(N,)
    dLSE = np.squeeze(dcross_entropy_loss) * 1

    #! backprop - LSE and x_exp -> ( LSE = np.log(np.sum(np.exp(x), axis=1)) )
    dx = dLSE[:, np.newaxis] * (x_exp / x_exp_sum)

    dx[range(N), y] += dscores_true * 1

    # #! backprop - softmax and x_exp, x_exp_sum -> (softmax = x_exp / x_exp_sum))
    # #? dx_exp(N, C) = dsoftmax(N, C) * (1. / x_exp_sum)(N, C: broadcasted)
    # #? dx_exp_sum(N, 1) = dsoftmax(N, C) * (x_exp / np.square(x_exp_sum))(N, C: broadcasted)
    # dx_exp = dsoftmax * (1. / x_exp_sum)
    # dx_exp_sum = np.sum(dsoftmax * -(x_exp / np.square(x_exp_sum)), axis=1, keepdims=True)

    # #! backprop - x_exp_sum and x_exp -> (x_exp_sum = np.sum(x_exp, axis=1))
    # #? dx_exp(N, C) = dx_exp_sum()
    # dx_exp += dx_exp_sum * np.ones_like(x_exp)

    # #! backprop - x_exp and x -> (exp)
    # #? dx(N, C) += x_exp(N, C) * dx_exp(N, C)
    # dx = dx_exp * x_exp

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #! Compute Minibatch stochastic
        #? sample_sum(D,) = np.sum(x(N, D), axis=0)
        #? sample_mean(D,) = x / sample_sum(N: broadcasted, D)
        #? sample_variance(D,) = np.sum((x - sample_mean)(N: broadcasted, D)**2, axis=0)(D,) / sample_sum(N: broadcasted, D)
        sample_sum = np.sum(x, axis=0, dtype=np.float64)
        sample_mean = sample_sum / N
        sample_var = np.sum((x - sample_mean)**2, axis=0) / N

        #! Update running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var 

        #! normalized x
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)

        #! batch normalization with shift and scale factor
        x_batch_normalized = x_normalized * gamma + beta
        out = x_batch_normalized

        # axis = bn_param.get('axis', 0)
        # cache = x, sample_mean, sample_var, np.sqrt(sample_var), gamma, x_normalized, (N, D), axis # save for backprop

        cache = {}
        cache['x'] = x
        cache['sample_sum'] = sample_sum
        cache['sample_mean'] = sample_mean
        cache['sample_var'] = sample_var
        cache['x_normalized'] = x_normalized
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['eps'] = eps
        cache['bn_param'] = bn_param

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        x_batch_normalized = x_normalized * gamma + beta
        out = x_batch_normalized

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #! load cached variables
    # x, x_mean, x_var, x_std, gamma, x_normalized, shape, axis = cache          # expand cache
    beta = cache['beta']
    gamma = cache['gamma']
    x_normalized = cache['x_normalized']
    N, D = x_normalized.shape
    x_var = cache['sample_var']
    x_mean = cache['sample_mean']
    x = cache['x']
    eps = cache['eps']
    # eps = 1e-5
    

    #! out(N, D) = gamma(N: broadcasted, D) * x_normalized(N, D) + beta(N: broadcasted, D)
    #? dbeta(D,) = np.sum(dout(N, D), axis=0)(squashed, D)
    #? dgamma(D,) = np.sum(dout(N, D) * x_normalzied(N, D), axis=0)(squashed, D)
    #? x_normalized(N, D) = dout(N, D) gamma(D,)
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    dx_normalized = dout * gamma

    #! x_normalized(N, D) = (x(N, D) - x_mean(N: broadcasted, D)) / np.sqrt(x_var(N: broadcasted, D))
    #? dx(N, D) = dx_normalized(N, D) / x_var(N: broadcasted, D)
    #? dx_mean(D,) = -1 * np.sum(dx_normalized(N, D) / x_var(N: broadcasted, D), axis=0)(squashed, D)
    #? dx_var(D,) = -1 * np.sum(dx_normalized(N, D) * (x(N, D) - x_mean(N: broadcasted, D)) / (x_var(N: broadcasted, D)**2), axis=0) (squashed, D)
    dx = dx_normalized / np.sqrt(x_var + eps)
    dx_mean = -1. * np.sum(dx_normalized / np.sqrt(x_var + eps), axis=0)
    dx_var = np.sum(dx_normalized * (x - x_mean) * -0.5 * (x_var + eps)**(-1.5), axis=0)

    #! x_var(D,) = np.sum((x - x_mean)**2, axis=0)(squashed, D) / N
    dx_mean += -2. * np.sum(dx_var * (x - x_mean))
    dx += 2*dx_var*(x - x_mean) / N


    #! x_mean(D,) = x_sum(D,) / N
    #? dx_sum(D,) = dx_mean / N
    dx_sum = dx_mean / N

    #! x_sum = np.sum(x, axis=0)
    dx += dx_sum

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # _, _, _, std, gamma, x_hat, shape, axis = cache # expand cache
    # S = lambda x: x.sum(axis=0)                     # helper function
    
    # dbeta = dout.reshape(shape, order='F').sum(axis)            # derivative w.r.t. beta
    # dgamma = (dout * x_hat).reshape(shape, order='F').sum(axis) # derivative w.r.t. gamma
    
    # dx = dout * gamma / (len(dout) * std)          # temporarily initialize scale value
    # dx = len(dout)*dx  - S(dx*x_hat)*x_hat - S(dx) # derivative w.r.t. unnormalized x

    # #! load cached variables
    beta = cache['beta']
    gamma = cache['gamma']
    x_normalized = cache['x_normalized']
    N, D = x_normalized.shape
    x_var = cache['sample_var']
    x_mean = cache['sample_mean']
    x = cache['x']
    eps = cache['eps']
    x_std = np.sqrt(x_var + eps)
    
    # #! out(N, D) = gamma(N: broadcasted, D) * x_normalized(N, D) + beta(N: broadcasted, D)
    # #? dbeta(D,) = np.sum(dout(N, D), axis=0)(squashed, D)
    # #? dgamma(D,) = np.sum(dout(N, D) * x_normalzied(N, D), axis=0)(squashed, D)
    # #? x_normalized(N, D) = dout(N, D) gamma(D,)

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    dx_normalized = dout * gamma

    dx = dx_normalized * (1 - 1./N - ((x-x_mean)**2)/(N*(x_var + eps))) / x_std

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = x.shape

    #! Compute Minibatch stochastic
    sample_sum = np.sum(x, axis=1, dtype=np.float64)
    sample_mean = sample_sum / D
    sample_var = np.sum((x - sample_mean[:, np.newaxis])**2, axis=1) / D

    #! normalized x
    x_normalized = (x - sample_mean[:, np.newaxis]) / np.sqrt(sample_var[:, np.newaxis] + eps)

    #! batch normalization with shift and scale factor
    x_batch_normalized = x_normalized * gamma + beta
    out = x_batch_normalized

    #! save to cache
    cache = {}
    cache['x'] = x
    cache['sample_sum'] = sample_sum
    cache['sample_mean'] = sample_mean
    cache['sample_var'] = sample_var
    cache['x_normalized'] = x_normalized
    cache['gamma'] = gamma
    cache['beta'] = beta
    cache['eps'] = eps
    cache['ln_param'] = ln_param

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  
    #! load cached variables
    beta = cache['beta']
    gamma = cache['gamma']
    x_normalized = cache['x_normalized']
    N, D = x_normalized.shape
    x_var = cache['sample_var'][:, np.newaxis]
    x_mean = cache['sample_mean'][:, np.newaxis]
    x = cache['x']
    eps = cache['eps']
    x_std = np.sqrt(x_var + eps)
    

    #! out(N, D) = gamma(N: broadcasted, D) * x_normalized(N, D) + beta(N: broadcasted, D)
    #? dbeta(D,) = np.sum(dout(N, D), axis=0)(squashed, D)
    #? dgamma(D,) = np.sum(dout(N, D) * x_normalzied(N, D), axis=0)(squashed, D)
    #? x_normalized(N, D) = dout(N, D) gamma(D,)
    axis=1
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    dx_normalized = dout * gamma

    #! x_normalized(N, D) = (x(N, D) - x_mean(N: broadcasted, D)) / np.sqrt(x_var(N: broadcasted, D))
    #? dx(N, D) = dx_normalized(N, D) / x_var(N: broadcasted, D)
    #? dx_mean(D,) = -1 * np.sum(dx_normalized(N, D) / x_var(N: broadcasted, D), axis=0)(squashed, D)
    #? dx_var(D,) = -1 * np.sum(dx_normalized(N, D) * (x(N, D) - x_mean(N: broadcasted, D)) / (x_var(N: broadcasted, D)**2), axis=0) (squashed, D)
    dx = dx_normalized / np.sqrt(x_var + eps)
    dx_mean = -1. * np.sum(dx_normalized / np.sqrt(x_var + eps), axis=axis)[:, np.newaxis]
    dx_var = np.sum(dx_normalized * (x - x_mean) * -0.5 * (x_var + eps)**(-1.5), axis=axis)[:, np.newaxis]

    #! x_var(D,) = np.sum((x - x_mean)**2, axis=0)(squashed, D) / N
    dx_mean += -2. * np.sum(dx_var * (x - x_mean))
    dx += 2*dx_var*(x - x_mean) / D


    #! x_mean(D,) = x_sum(D,) / N
    #? dx_sum(D,) = dx_mean / N
    dx_sum = dx_mean / D

    #! x_sum = np.sum(x, axis=0)
    dx += dx_sum

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = np.random.rand(*x.shape) < p
        out = x * mask / p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # ! only gradient flow on the neurons that survive at the test time
        # ? mask = np.random.rand(*x.shape) < p
        # ? out = x * mask
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modifiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ! load variables
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    # ! pad to only (H, W) axes
    x_padded = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    _, _, H_padded, W_padded = x_padded.shape
    # print('x_padded shape: ', x_padded.shape)

    # ! output
    H_out = 1 + (H + 2*pad - HH) // stride
    W_out = 1 + (W + 2*pad - WW) // stride

    # Create output tensor
    out = np.zeros((N, F, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            top = i * stride
            left = j * stride

            # ! modify dimension
            # ? x_slice: (N, C, HH, WW)
            # ? x_slice_expanded: (N, 1, C, HH, WW)
            # ? w_expanded: (1, F, C, HH, WW)
            x_slice = x_padded[:, :, top:top+HH, left:left+WW]
            x_slice_expanded = x_slice[:, None, ...]
            w_expanded = w[None, ...]

            # ! convolution
            # ? (N, F) = np.sum(x_slice_expanded(N, 1, C, HH, WW) * w_expanded(1, F, C, HH, WW), axis=(2, 3, 4)) 
            convolution = np.sum(x_slice_expanded * w_expanded, axis=(2, 3, 4)) + b
            out[:, :, i, j] = convolution
            

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ! unpacking variables
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # ! 
    x_padded = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    _, _, H_padded, W_padded = x_padded.shape
    print('x_padded shape: ', x_padded.shape)

    H_out = 1 + (H + 2*pad - HH) // stride
    W_out = 1 + (W + 2*pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))
    print('out shape: ', out.shape) 

    # ! dx, dw, db initialize
    dx = np.zeros_like(x)
    # dw = np.zeros_like(w)
    # db = np.zeros_like(b)

    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)

    for i in range(H_out):
        for j in range(W_out):
            top = i * stride
            left = j * stride

            # ! expand dimension for broadcasting
            # ? dout: (N, F, H_out, W_out)
            # ? dout_slice: (N, F, 1, 1, 1)
            # ? x_slice: (N, C, HH, WW)
            # ? x_slice_expanded: (N, 1, C, HH, WW)
            # ? w_expanded: (1, F, C, HH, WW)
            dout_slice_expanded = dout[..., i, j][..., None, None, None] # (N, F, 1, 1, 1)
            x_slice = x_padded[..., top:top+HH, left:left+WW] # (N, C, HH, WW)
            x_slice_expanded = x_slice[:, None, ...] # (N, 1, C, HH, WW)

            # * collapse to N axis using np.sum(axis=0)
            dw += np.sum(dout_slice_expanded * x_slice_expanded, axis=0) # (N, F, C, HH, WW) -> (F, C, HH, WW)

            # * collapse to F axis using np.sum(axis=1)
            w_expanded = w[None, ...] # (1, F, C, HH, WW)
            dx_padded[..., top:top+HH, left:left+WW] \
                += np.sum(dout_slice_expanded * w_expanded, axis=1) # (N, F, C, HH, WW) -> (N, C, HH, WW)

    dx = dx_padded[..., pad:-pad, pad:-pad] # delete padding

    # * maintain only F axis => collapse on (N, HH, WW)
    db = np.sum(dout, axis=(0, 2, 3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ! unpack variables
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out))

    cache={}
    max_indices_arr = np.zeros((H_out, W_out, N, C), dtype=np.int32)
    for h in range(H_out):
        for w in range(W_out):
            top = h * stride
            left = w * stride
            x_slice = x[:, :, top:top+pool_height, left:left+pool_width] # (N, C, p_h, p_w)
            x_slice_flatten = x_slice.reshape(N, C, -1) # (N, C, 1)

            max_indices = np.argmax(x_slice_flatten, axis=2) # (N, C)
            max_indices_arr[h, w] = max_indices
            max_row, max_column = np.unravel_index(max_indices, (pool_height, pool_width)) # (N, C), (N, C)
            nc_indices = np.meshgrid(range(N), range(C), indexing='ij')

            out[nc_indices[0], nc_indices[1], h, w] = \
                x_slice[
                    nc_indices[0], 
                    nc_indices[1], 
                    max_row[nc_indices[0], nc_indices[1]],
                    max_column[nc_indices[0], nc_indices[1]]
                ]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, max_indices_arr)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ! unpack variables
    # x, pool_param = cache
    x, pool_param, max_indices_arr = cache

    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    dx = np.zeros_like(x)
    for h in range(H_out):
        for w in range(W_out):
            top = h * stride
            left = w * stride
            dx_slice = dx[:, :, top:top+pool_height, left:left+pool_width] # (N, C, p_h, p_w)
            # x_slice = x[:, :, top:top+pool_height, left:left+pool_width] # (N, C, p_h, p_w)
            # x_slice_flatten = x_slice.reshape(N, C, -1) # (N, C, 1)

            # max_indices = np.argmax(x_slice_flatten, axis=2) # (N, C)
            max_indices = max_indices_arr[h, w]
            max_row, max_column = np.unravel_index(max_indices, (pool_height, pool_width)) # (N, C), (N, C)
            nc_indices = np.meshgrid(range(N), range(C), indexing='ij')

            dx_slice[
                nc_indices[0],
                nc_indices[1],
                max_row[nc_indices[0], nc_indices[1]],
                max_column[nc_indices[0], nc_indices[1]]
                ] = dout[nc_indices[0], nc_indices[1], h, w]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape

    channels_per_group = C // G 

    g_indices = [
        slice(i * channels_per_group, (i + 1) * channels_per_group) for i in range(G)
    ]

    cache=[]
    cache.append(G)
    out = np.zeros_like(x)
    for g in g_indices:
        # print(g.start, g.stop)
        x_sliced = x[:, g.start:g.stop, :, :]
        gamma_sliced = gamma[:, g.start:g.stop, :, :]
        beta_sliced = beta[:, g.start:g.stop, :, :]

        x_sliced_transposed = x_sliced.transpose(0, 2, 3, 1)
        x_sliced_transposed_reshaped = x_sliced_transposed.reshape(-1, channels_per_group)
        out_tmp, cache_tmp = layernorm_forward(
                                    x_sliced_transposed_reshaped,
                                    gamma_sliced.squeeze(),
                                    beta_sliced.squeeze(),
                                    gn_param)
        out[:, g.start:g.stop, :, :] = out_tmp.reshape(x_sliced_transposed.shape).transpose(0, 3, 1, 2)
        cache.append(cache_tmp)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    _, C, _, _ = dout.shape
    G = cache[0]
    channels_per_group = C // G 

    g_indices = [
        slice(i * channels_per_group, (i + 1) * channels_per_group) for i in range(G)
    ] 

    dx = np.zeros_like(dout)
    dgamma = np.zeros((1, C, 1, 1))
    dbeta = np.zeros((1, C, 1, 1))
    for idx, g in enumerate(g_indices):
        dout_sliced = dout[:, g.start:g.stop, :, :].transpose(0, 2, 3, 1)
        dout_sliced_reshaped = dout_sliced.reshape(-1, channels_per_group)

        dx_tmp, dgamma_tmp, dbeta_tmp = layernorm_backward(dout_sliced_reshaped, cache[idx+1])
        dx[:, g.start:g.stop, :, :] = dx_tmp.reshape(dout_sliced.shape).transpose(0, 3, 1, 2)
        dgamma[:, g.start:g.stop, :, :] = dgamma_tmp.reshape(1, *dgamma_tmp.shape, 1, 1)
        dbeta[:, g.start:g.stop, :, :] = dbeta_tmp.reshape(1, *dbeta_tmp.shape, 1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
