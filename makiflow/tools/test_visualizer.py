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

import matplotlib.pyplot as plt
import seaborn as sns


class TestVisualizer:
    @staticmethod
    def plot_test_values(test_values, legends, x_label, y_label, save_path=None):
        assert(len(test_values) == len(legends))
        # Create error graphs
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)

        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        for i in range(len(test_values)):
            axes.plot(test_values[i], label=legends[i])
        axes.legend()

        axes.set_ylabel(y_label)
        axes.set_xlabel(x_label)

        if save_path is not None:
            fig.savefig(save_path)
        if save_path is None:
            fig.show()
        plt.close(fig)

    @staticmethod
    def plot_numpy_dist_obs(values, legends, save_path=None, x_axis=None, y_axis=None):
        """
        

        Parameters
        ----------
        values : list
            A set of numpy.ndarray each one of them have an arbitrary shape.
            For each of `values` distribution will be plotted.
        legends : list
            List of values names.
        save_path : str
            Path to save plotted figure.
        x_axis : tuple
            Tuple of (min, max), where max and min are the maximum and minimum values on the X axis accordingly.
            If set to None, maximum and minimum values are not constrained.
        y_axis : tuple
            Tuple of (min, max), where max and min are the maximum and minimum values on the Y axis accordingly.
            If set to None, maximum and minimum values are not constrained.
        """
        assert (len(values) == len(legends))
        fig = plt.figure(figsize=(6, len(values) * 6))

        for i in range(len(values)):
            sub = plt.subplot(len(values),1,i+1)
            plt.title(legends[i] + f' number {i}')
            axes = sub.axes
            if y_axis is not None:
                axes.set_ylim(y_axis)
            if x_axis is not None:
                axes.set_xlim(x_axis)
            sns.distplot(values[i])

        if save_path is not None:
            fig.savefig(save_path)
        else:
            fig.show()

        plt.close(fig)

