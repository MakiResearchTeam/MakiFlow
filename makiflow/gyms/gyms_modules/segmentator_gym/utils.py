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
import os
from makiflow.gyms.gyms_modules.gyms_collector import GymCollector, SEGMENTATION, TESTER
from makiflow.gyms.core import TesterBase
from makiflow.tools.preprocess import preprocess_input
from makiflow.metrics import categorical_dice_coeff
from makiflow.metrics import confusion_mat
from makiflow.tools.test_visualizer import TestVisualizer
from sklearn.metrics import f1_score
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_heatmap(heatmap, name_heatmap=None, shift_image=60, dpi=80):
    h, w = heatmap.shape

    figsize = w / float(dpi), h / float(dpi)

    fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    sns.heatmap(heatmap)
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = np.reshape(data, (h, w, 3))

    plt.close('all')
    """
    if name_heatmap is None:
        return data.astype(np.uint8, copy=False)
    else:
        return put_text_on_image(data.astype(np.uint8, copy=False), name_heatmap, shift_image=shift_image)
    """
    return data.astype(np.uint8, copy=False)


def put_text_on_image(image, text, shift_image=60, medium_size=800):
    h, w = image.shape[:-1]
    img = (np.ones((h + shift_image, w, 3)) * 255.0).astype(image.dtype, copy=False)
    img[:h] = image

    cv2.putText(
        img,
        text,
        (shift_image // 4, h + shift_image // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        min(h / medium_size, w / medium_size),
        (0, 0, 0),
        1
    )

    return img.astype(image.dtype, copy=False)

