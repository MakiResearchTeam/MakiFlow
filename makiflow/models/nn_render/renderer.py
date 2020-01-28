import tensorflow as tf
from makiflow.base.maki_entities import MakiModel


class NeuralRenderer(MakiModel):
    def __init__(self, input_l, output, name):
        self.name = str(name)
        graph_tensors = output.get_previous_tensors()
        graph_tensors.update(output.get_self_pair())
        super().__init__(graph_tensors, outputs=[output], inputs=[input_l])

    def predict(self, x):
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def _get_model_info(self):
        return {
            'name': self.name,
            'input_s': self._inputs[0].get_name(),
            'output': self._outputs[0].get_name()
        }

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------SETTING UP TRAINING-----------------------------------------

    def _prepare_training_vars(self):
        out_shape = self._outputs[0].get_shape()
        self.out_h = out_shape[1]
        self.out_w = out_shape[2]
        self.batch_sz = out_shape[0]

        self._uv_maps = self._input_data_tensors[0]
        self._images = tf.placeholder(tf.int32, shape=out_shape, name='images')

        self._training_out = self._training_outputs[0]

        self._training_vars_are_ready = True

        self._mse_loss_is_build = False
        self._abs_loss_is_build = False

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------ABS LOSS----------------------------------------------------

    def _build_abs_loss(self):
        diff = tf.abs(self._training_out - self._images)
        self._abs_loss = tf.reduce_mean(diff)
        self._final_abs_loss = self._build_final_loss(self._abs_loss)

    def _setup_abs_loss_inputs(self):
        pass

    def _minimize_abs_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._abs_loss_is_build:
            self._setup_abs_loss_inputs()
            self._build_abs_loss()
            self._abs_optimizer = optimizer
            self._abs_train_op = optimizer.minimize(
                self._final_abs_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._abs_optimizer != optimizer:
            print('New optimizer is used.')
            self._abs_optimizer = optimizer
            self._abs_train_op = optimizer.minimize(
                self._final_abs_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._abs_train_op

