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
    def plot_numpy_dist_obs(values, legends, save_path=None, axes_x=None, axes_y=None):
        """
        Parameters
        ----------
        values : list
            Set of values that need to plot.
        legends : list
            List of values names.
        save_path : str
            Path to save plotted figure.
        axes_x : tuple
            Tuple of size 2, example (min_x, max_x), where min_x - minimum on the axes x, max_x - maximum of the axes x.
            By default is None, mean that it will be self-scaled.
        axes_y : tuple
            Tuple of size 2, example (min_y, max_y), where min_y - minimum on the axes y, max_x - maximum of the axes y.
            By default is None, mean that it will be self-scaled.
        """
        assert (len(values) == len(legends))
        fig = plt.figure(figsize=(6, 34))

        for i in range(len(values)):
            sub = plt.subplot(len(values),1,i+1)
            plt.title(legends[i] + f' number {i}')
            axes = sub.axes
            if axes_y is not None:
                axes.set_ylim(axes_y)
            if axes_x is not None:
                axes.set_xlim(axes_x)
            sns.distplot(values[i])

        if save_path is not None:
            fig.savefig(save_path)
        else:
            fig.show()

        fig.close()

