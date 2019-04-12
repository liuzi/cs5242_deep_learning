import numpy as np

# Attension:
# - Never change the value of input, which will change the result of backward


class operation(object):
    """
    Operation abstraction
    """

    def forward(self, input):
        """Forward operation, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward operation, return gradient to input"""
        raise NotImplementedError


class relu(operation):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, out_grad, input):
        in_grad = (input >= 0) * out_grad
        return in_grad


class flatten(operation):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, input):
        batch = input.shape[0]
        output = input.copy().reshape(batch, -1)
        return output

    def backward(self, out_grad, input):
        in_grad = out_grad.copy().reshape(input.shape)
        return in_grad


class matmul(operation):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, input, weights):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        return np.matmul(input, weights)

    def backward(self, out_grad, input, weights):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            in_grad: gradient to the forward input with same shape as input
            w_grad: gradient to weights, with same shape as weights            
        """
        in_grad = np.matmul(out_grad, weights.T)
        w_grad = np.matmul(input.T, out_grad)
        return in_grad, w_grad


class add_bias(operation):
    def __init__(self):
        super(add_bias, self).__init__()

    def forward(self, input, bias):
        '''
        # Arugments
          input: numpy array with shape (batch, in_features)
          bias: numpy array with shape (in_features)

        # Returns
          output: numpy array with shape(batch, in_features)
        '''
        return input + bias.reshape(1, -1)

    def backward(self, out_grad, input, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            bias: numpy array with shape (out_features)
        # Returns
            in_grad: gradient to the forward input with same shape as input
            b_bias: gradient to bias, with same shape as bias
        """
        in_grad = out_grad
        b_grad = np.sum(out_grad, axis=0)
        return in_grad, b_grad


class fc(operation):
    def __init__(self):
        super(fc, self).__init__()
        self.matmul = matmul()
        self.add_bias = add_bias()

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        output = self.matmul.forward(input, weights)
        output = self.add_bias.forward(output, bias)
        # output = np.matmul(input, weights) + bias.reshape(1, -1)
        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            in_grad: gradient to the forward input of fc layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        # in_grad = np.matmul(out_grad, weights.T)
        # w_grad = np.matmul(input.T, out_grad)
        # b_grad = np.sum(out_grad, axis=0)
        out_grad, b_grad = self.add_bias.backward(out_grad, input, bias)
        in_grad, w_grad = self.matmul.backward(out_grad, input, weights)
        return in_grad, w_grad, b_grad


def im2col(input, kernel_h, kernel_w, stride, pad):
    N, C, H, W = input.shape
    output_h = (H + 2 * pad - kernel_h) // stride + 1
    output_w = (W + 2 * pad - kernel_w) // stride + 1

    input_padded = np.pad(input, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, kernel_h, kernel_w, output_h, output_w))

    for h in range(kernel_h):
        h_max = h + stride * output_h
        for w in range(kernel_w):
            w_max = w + stride * output_w
            col[:, :, h, w, :, :] = input_padded[:, :, h:h_max:stride, w:w_max:stride]

    col_input = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * output_h * output_w, -1)
    return col_input, N, output_h, output_w


def col2im(d_col, input_shape, kernel_h, kernel_w, stride, pad):
    N,C,H,W = input_shape
    output_h = (H + 2*pad - kernel_h)//stride + 1
    output_w = (W + 2*pad - kernel_w)//stride + 1
    col = d_col.reshape(N, output_h, output_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    in_grad = np.zeros((N, C, H, W))
    for h in range(kernel_h):
        h_max = h +stride*output_h
        for w in range(kernel_w):
            w_max = w + stride*output_w
            in_grad[:, :, h:h_max:stride, w:w_max:stride] += col[:, :, h, w, :, :]
    return in_grad


class conv(operation):
    def __init__(self, conv_params):
        """
        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad = 2 means a 2-pixel border of padded with zeros
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
        """
        super(conv, self).__init__()
        self.conv_params = conv_params
        self.col_input = None
        self.col_kernel = None

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            output: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        output = None

        #########################################
        # TODO:code here
        # N,_,H,W = input.shape
        # output_h=(H+2*pad-kernel_h)//stride+1
        # output_w=(W+2*pad-kernel_w)//stride+1
        #
        # input_padded = np.pad(input, [(0,0),(0,0),(pad,pad),(pad,pad)], 'constant')
        # col = np.zeros((N, in_channel, kernel_h, kernel_w, output_h, output_w))
        #
        # for h in range(kernel_h):
        #     h_max = h + stride*output_h
        #     for w in range(kernel_w):
        #         w_max = w+stride*output_w
        #         col[:,:,h,w,:,:] = input_padded[:, :, h:h_max:stride, w:w_max:stride]
        #
        # col_input = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*output_h*output_w, -1)
        col_input, N, output_h, output_w = im2col(input, kernel_h, kernel_w, stride, pad)
        col_kernel = weights.reshape(out_channel, -1).T

        self.col_input = col_input
        self.col_kernel = col_kernel

        output = np.matmul(col_input, col_kernel) + bias
        output = output.reshape(N, output_h, output_w, -1).transpose(0,3,1,2)

        #########################################

        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, out_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            in_grad: gradient to the forward input of conv layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        in_grad = None
        w_grad = None
        b_grad = None

        #########################################
        # TODO: code here
        out_grad=out_grad.transpose(0, 2, 3, 1).reshape(-1, out_channel)
        b_grad = np.sum(out_grad, axis=0, keepdims=True)
        w_grad = np.matmul(self.col_input.T, out_grad)
        w_grad = w_grad.transpose(1,0).reshape(out_channel,in_channel,kernel_h, kernel_w)

        d_col = np.matmul(out_grad, self.col_kernel.T)
        in_grad = col2im(d_col, input.shape, kernel_h, kernel_w, stride, pad)

        # N,C,H,W = input.shape
        # output_h = (H + 2*pad - kernel_h)//stride + 1
        # output_w = (W + 2*pad - kernel_w)//stride + 1
        # col = d_col.reshape(N, output_h, output_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
        # in_grad = np.zeros((N, C, H, W))
        # for h in range(kernel_h):
        #     h_max = h +stride*output_h
        #     for w in range(kernel_w):
        #         w_max = w + stride*output_w
        #         in_grad[:, :, h:h_max:stride, w:w_max:stride] += col[:, :, h, w, :, :]

        #########################################

        return in_grad, w_grad, b_grad


class pool(operation):
    def __init__(self, pool_params):
        """
        # Arguments
            pool_params: dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad = 2 means a 2-pixel border of padding with zeros.
        """
        super(pool, self).__init__()
        self.pool_params = pool_params
        self.arg_max = None

    def forward(self, input):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            output: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        output = None

        #########################################
        # TODO: code here
        col, N, output_h, output_w = im2col(input, pool_height, pool_width, stride, pad)
        col = col.reshape(-1, pool_height*pool_width)
        if pool_type == 'max':
            self.arg_max = np.argmax(col, axis=1)
            output = np.max(col, axis=1)
        elif pool_type == 'avg':
            output = np.average(col, axis=1)
        else:
            raise ValueError('Doesn\'t support \'%s\' pooling.' %
                             pool_type)
        output = output.reshape(N, output_h, output_w, input.shape[1]).transpose(0, 3, 1, 2)
        #########################################
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, in_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            in_grad: gradient to the forward input of pool layer, with same shape as input
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        in_grad = None

        #########################################
        # TODO: code here
        out_grad = out_grad.transpose(0, 2, 3, 1)
        pool_size = pool_height*pool_width
        if pool_type == 'max':
            in_grad = np.zeros((out_grad.size, pool_size))
            in_grad[np.arange(self.arg_max.size), self.arg_max.flatten()] = out_grad.flatten()
            in_grad = in_grad.reshape(out_grad.shape + (pool_size,))
        elif pool_type == 'avg':
            avg_out_grad = out_grad.flatten / out_grad.size
            avg_grad = np.zeros((out_grad.size, pool_size))
            in_grad[range(out_grad.size), :] = avg_out_grad
        else:
            raise ValueError('Doesn\'t support \'%s\' pooling.' %
                             pool_type)
        grad_col = in_grad.reshape(in_grad.shape[0]*in_grad.shape[1]*in_grad.shape[2], -1)
        in_grad = col2im(grad_col, input.shape, pool_height, pool_width, stride, pad)
        #########################################

        return in_grad


class dropout(operation):
    def __init__(self, rate, training=True, seed=None):
        """
        # Arguments
            rate: float[0, 1], the probability of setting a neuron to zero
            training: boolean, apply this layer for training or not. If for training, randomly drop neurons, else DO NOT drop any neurons
            seed: int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input
        """
        self.rate = rate
        self.seed = seed
        self.training = training
        self.mask = None

    def forward(self, input):
        """
        # Arguments
            input: numpy array with any shape

        # Returns
            output: same shape as input
        """
        output = None
        if self.training:
            np.random.seed(self.seed)
            p = np.random.random_sample(input.shape)
            #########################################
            # TODO: code here
            self.mask = p > self.rate
            output = input * self.mask / (1-p)
            #########################################
        else:
            output = input
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to forward output of dropout, same shape as input
            input: numpy array with any shape
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input

        # Returns
            in_grad: gradient to forward input of dropout, same shape as input
        """
        if self.training:
            #########################################
            # TODO: code here
            in_grad = None
            np.random.seed(self.seed)
            p = np.random.random_sample(input.shape)
            in_grad = out_grad * self.mask / (1-p)
            #########################################
        else:
            in_grad = out_grad
        return in_grad


class softmax_cross_entropy(operation):
    def __init__(self):
        super(softmax_cross_entropy, self).__init__()

    def forward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            output: scalar, average loss
            probs: the probability of each category
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)

        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)
        output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch
        return output, probs

    def backward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            in_grad: gradient to forward input of softmax cross entropy, with shape (batch, num_class)
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)
        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)

        in_grad = probs.copy()
        in_grad[np.arange(batch), labels] -= 1
        in_grad /= batch
        return in_grad
