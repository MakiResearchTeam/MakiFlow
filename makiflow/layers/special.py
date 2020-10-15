# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf
from makiflow.core import MakiRestorable, MakiLayer


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


class LogMelSpectrogramLayer(MakiLayer):
    TYPE = 'LogMelSpectrogramLayer'
    SAMPLE_RATE = 'SAMPLE_RATE'
    FFT_SIZE = 'FFT_SIZE'
    HOP_SIZE = 'HOP_SIZE'
    N_MELS = 'N_MELS'
    F_MIN = 'F_MIN'
    F_MAX = 'F_MAX'

    @staticmethod
    def build(params: dict):
        sample_rate = params[LogMelSpectrogramLayer.SAMPLE_RATE]
        name = params[MakiRestorable.NAME]
        fft_size = params[LogMelSpectrogramLayer.FFT_SIZE]
        hop_size = params[LogMelSpectrogramLayer.HOP_SIZE]
        n_mels = params[LogMelSpectrogramLayer.N_MELS]
        f_min = params[LogMelSpectrogramLayer.F_MIN]
        f_max = params[LogMelSpectrogramLayer.F_MAX]
        return LogMelSpectrogramLayer(
            sample_rate=sample_rate,
            name=name,
            fft_size=fft_size,
            hop_size=hop_size,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )

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
