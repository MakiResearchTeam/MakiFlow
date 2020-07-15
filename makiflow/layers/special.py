import tensorflow as tf
from .sf_layer import SimpleForwardLayer
from makiflow.base.maki_entities.maki_layer import MakiRestorable


def _tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def _power_to_db(magnitude, amin=1e-16, top_db=80):
    ref_value = tf.reduce_max(magnitude)
    log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
    return log_spec


class LogMelSpectrogramLayer(SimpleForwardLayer):
    TYPE = 'LogMelSpectrogramLayer'
    SAMPLE_RATE = 'SAMPLE_RATE'
    FFT_SIZE = 'FFT_SIZE'
    HOP_SIZE = 'HOP_SIZE'
    N_MELS = 'N_MELS'
    F_MIN = 'F_MIN'
    F_MAX = 'F_MAX'

    def __init__(self, sample_rate, name, fft_size=1024, hop_size=512, n_mels=128,
                 f_min=0.0, f_max=None):
        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._hop_size = hop_size
        self._n_mels = n_mels
        self._f_min = f_min
        self._f_max = f_max if f_max else sample_rate / 2
        self._mel_filter_bank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self._n_mels,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self._sample_rate,
            lower_edge_hertz=self._f_min,
            upper_edge_hertz=self._f_max
        )
        super().__init__(
            name=name,
            params=[],
            regularize_params=[],
            named_params_dict={}
        )

    def _forward(self, x):
        spectrograms = tf.signal.stft(
            x,
            frame_length=self._fft_size,
            frame_step=self._hop_size,
            pad_end=False
        )
        magnitude_spectrograms = tf.abs(spectrograms)  # magnitudes of complex numbers
        sq_magnitude_spectrograms = tf.square(magnitude_spectrograms)
        mel_spectrograms = tf.matmul(sq_magnitude_spectrograms, self._mel_filter_bank)
        log_mel_spectrograms = _power_to_db(mel_spectrograms)
        return log_mel_spectrograms

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: LogMelSpectrogramLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self._name,
                LogMelSpectrogramLayer.SAMPLE_RATE: self._sample_rate,
                LogMelSpectrogramLayer.FFT_SIZE: self._fft_size,
                LogMelSpectrogramLayer.HOP_SIZE: self._hop_size,
                LogMelSpectrogramLayer.N_MELS: self._n_mels,
                LogMelSpectrogramLayer.F_MIN: self._f_min,
                LogMelSpectrogramLayer.F_MAX: self._f_max
            }
        }
