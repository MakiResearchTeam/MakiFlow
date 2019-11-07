import numpy as np

# Some initializate methods
# Initializations define the way to set the initial random weights of MakiFlow layers.
def init_conv_kernel(kw, kh, in_f, out_f, kernel_initializer):
    W = np.random.randn(kw, kh, in_f, out_f)
    if kernel_initializer == 'xavier_gaussian_avg':
        W *= np.sqrt(3. / (kw * kh * in_f + kw * kh * out_f))

    elif kernel_initializer == 'xavier_gaussian_inf':
        W *= np.sqrt(1. / (kw * kh * in_f))

    elif kernel_initializer == 'xavier_uniform_avg':
        W = np.random.uniform(low=-1., high=1.0, size=[kw, kh, in_f, out_f])
        W *= np.sqrt(6. / (kw * kh * in_f + kw * kh * out_f))

    elif kernel_initializer == 'xavier_uniform_inf':
        W = np.random.uniform(low=-1., high=1.0, size=[kw, kh, in_f, out_f])
        W *= np.sqrt(3. / (kw * kh * in_f))

    elif kernel_initializer == 'he':
        W *= np.sqrt(2. / (kw * kh * in_f))

    elif kernel_initializer == 'lasange':
        W = np.random.uniform(low=-1., high=1.0, size=[kw, kh, in_f, out_f])
        W *= np.sqrt(12. / (kw * kh * in_f + kw * kh * out_f))

    return W.astype(np.float32)


def init_dense_mat(in_d, out_d, mat_initializer):
    W = np.random.randn(in_d, out_d)
    if mat_initializer == 'xavier_gaussian':
        W *= np.sqrt(3. / (in_d + out_d))

    elif mat_initializer == 'xavier_uniform':
        W = np.random.uniform(low=-1., high=1.0, size=[in_d, out_d])
        W *= np.sqrt(6. / (in_d + out_d))

    elif mat_initializer == 'he':
        W *= np.sqrt(2. / (in_d))

    elif mat_initializer == 'lasange':
        W = np.random.uniform(low=-1., high=1.0, size=[in_d, out_d])
        W *= np.sqrt(12. / (in_d + out_d))

    return W.astype(np.float32)
