Model code organizing
=====================

Each model consists of the following components:
    - main entity
    - training modules

The model package is organized as follows:
    - main_modules
        - main_module1
        - main_module2
    - training_modules
        - training_module1
        - training_module2
    - compile.py - compiles the main modules with the training ones and provides the full model.

Main modules
------------

Each model has the main class entity that provides the basic functionality for working with the model.
It also share the common components that are using by the training modules.

Interface of the main module class:

.. code-block:: python

    class ModelBasis(MakiModel):
        def __init__(self, ...)
            # Setting up the variables.
            self._training_vars_are_ready = False
            pass

        def _get_model_info(self):
            # This method is required by the MakiModel interface.
            pass

        # COMMON TRAINING FUNCTIONALITY
        def _prepare_training_vars(self):
            # Setting up the variables, losses, etc
            self._training_vars_are_ready = True
            pass

        def _other_methods_for_the_training_modules(self):
            pass

Training modules
----------------

Each training module is responsible for training the model using a certain loss function.
Therefore, its name reflects the employed training loss: LossNameTrainingModule.

Interface of the training module class:

.. code-block:: python

    class LossNameTrainingModule(ModelBasis):
        def _prepare_training_vars(self):
            self._lossname_loss_is_built = False
            super()._prepare_training_vars()

        def _build_lossname_loss(self):
            # Code that builds the scalar tensor of the minimized loss.
            lossname_loss = ...
            # This is the method built into the MakiModel.
            # It is used to include the regularization term into the total loss.
            self._final_lossname_loss = self._build_final_loss(lossname_loss)

        def _setup_lossname_inputs(self):
            # Here the necessary placeholder are set up.
            pass

        # This method signature can include other arguments if needed.
        def _minimize_lossname_loss(self, optimizer, global_step):
            if not self._training_vars_are_ready:
                self._prepare_training_vars()

            if not self._lossname_is_built:
                self._setup_lossname_inputs()
                self._build_lossname_loss()
                self._lossname_optimizer = optimizer
                self._lossname_train_op = optimizer.minimize(
                    self._final_lossname_loss, var_list=self._trainable_vars, global_step=global_step
                )
                self._session.run(tf.variables_initializer(optimizer.variables()))
                self._lossname_loss_is_built = True
                # This is a common utility for printing info messages
                loss_is_built()

            if self._lossname_optimizer != optimizer:
                # This is a common utility for printing info messages
                new_optimizer_used()
                self._lossname_optimizer = optimizer
                self._lossname_train_op = optimizer.minimize(
                    self._final_lossname_loss, var_list=self._trainable_vars, global_step=global_step
                )
                self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._lossname_train_op

        def fit_lossname(self, ..., optimizer, epochs=1, global_step=None):
            assert (optimizer is not None)
            assert (self._session is not None)

            train_op = self._minimize_abs_loss(optimizer, global_step)
            # Training cycle

You can copy this code a modify accordingly.

compile.py
----------

In this file all the modules are assembled into the final model.

.. code-block:: python

    from .training_modules import Lossname1TrainingModule, Lossname2TrainingModule


    class Model(Lossname1TrainingModule, Lossname2TrainingModule):
        pass

This model is then used for the one's purposes.
