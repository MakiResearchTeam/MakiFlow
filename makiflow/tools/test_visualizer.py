import matplotlib.pyplot as plt


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