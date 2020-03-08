import numpy as np

# Some initialize methods
# Initializations define the way to set the initial random weights of MakiFlow layers.

class InitConvKernel:

    XAVIER_GAUSSIAN_AVG = 'xavier_gaussian_avg'
    XAVIER_GAUSSIAN_INF = 'xavier_gaussian_inf'
    XAVIER_UNIFORM_INF = 'xavier_uniform_inf'
    XAVIER_UNIFORM_AVG = 'xavier_uniform_avg'
    LASANGE = 'lasange'
    HE = 'he'

    @staticmethod
    def init_by_name(kw, kh, out_f, in_f, name_init, dtype=np.float32):
        return InitConvKernel.SET_INITS[name_init](kw, kh, out_f, in_f, dtype)

    @staticmethod
    def xavier_gaussian_avg(kw, kh, out_f, in_f, dtype=np.float32):
        w = np.random.randn(kw, kh, in_f, out_f)
        w *= np.sqrt(3. / (kw * kh * in_f + kw * kh * out_f))
        return w.astype(dtype)

    @staticmethod
    def xavier_gaussian_inf(kw, kh, out_f, in_f, dtype=np.float32):
        w = np.random.randn(kw, kh, in_f, out_f)
        w *= np.sqrt(1. / (kw * kh * in_f))
        return w.astype(dtype)

    @staticmethod
    def xavier_uniform_avg(kw, kh, out_f, in_f, dtype=np.float32):
        w = np.random.randn(kw, kh, in_f, out_f)
        w *= np.sqrt(6. / (kw * kh * in_f + kw * kh * out_f))
        return w.astype(dtype)

    @staticmethod
    def xavier_uniform_inf(kw, kh, out_f, in_f, dtype=np.float32):
        w = np.random.randn(kw, kh, in_f, out_f)
        w *= np.sqrt(3. / (kw * kh * in_f))
        return w.astype(dtype)

    @staticmethod
    def he(kw, kh, out_f, in_f, dtype=np.float32):
        w = np.random.randn(kw, kh, in_f, out_f)
        w *= np.sqrt(2. / (kw * kh * in_f))
        return w.astype(dtype)

    @staticmethod
    def lasange(kw, kh, out_f, in_f, dtype=np.float32):
        w = np.random.randn(kw, kh, in_f, out_f)
        w *= np.sqrt(12. / (kw * kh * in_f + kw * kh * out_f))
        return w.astype(dtype)

    SET_INITS = {
            XAVIER_GAUSSIAN_AVG: xavier_gaussian_avg,
            XAVIER_GAUSSIAN_INF: xavier_gaussian_inf,

            XAVIER_UNIFORM_INF: xavier_uniform_inf,
            XAVIER_UNIFORM_AVG: xavier_uniform_avg,

            LASANGE: lasange,
            HE: he,
    }

class InitDenseMat:

    XAVIER_GAUSSIAN = 'xavier_gaussian'
    XAVIER_UNIFORM = 'xavier_uniform'
    HE = 'he'
    LASANGE = 'lasange'

    @staticmethod
    def init_by_name(in_d, out_d, name_init, dtype=np.float32):
        return InitDenseMat.SET_INITS[name_init](in_d, out_d, dtype)

    @staticmethod
    def xavier_gaussian(in_d, out_d, dtype=np.float32):
        w = np.random.randn(in_d, out_d)
        w *= np.sqrt(3. / (in_d + out_d))
        return w.astype(dtype)

    @staticmethod
    def xavier_uniform(in_d, out_d, dtype=np.float32):
        w = np.random.uniform(low=-1., high=1.0, size=[in_d, out_d])
        w *= np.sqrt(6. / (in_d + out_d))
        return w.astype(dtype)

    @staticmethod
    def he(in_d, out_d, dtype=np.float32):
        w = np.random.randn(in_d, out_d)
        w *= np.sqrt(2. / (in_d))
        return w.astype(dtype)

    @staticmethod
    def lassange(in_d, out_d, dtype=np.float32):
        w = np.random.uniform(low=-1., high=1.0, size=[in_d, out_d])
        w *= np.sqrt(12. / (in_d + out_d))
        return w.astype(dtype)

    SET_INITS = {
            XAVIER_GAUSSIAN: xavier_gaussian,
            XAVIER_UNIFORM: xavier_uniform,
            HE: he,
            LASANGE: lassange,
    }


