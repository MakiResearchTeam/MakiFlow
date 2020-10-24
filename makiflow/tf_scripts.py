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

LOW_MEMORY_SESS = None
FRACTION_MEMORY_SESS = None


def set_main_gpu(gpu_id):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    del os


def get_low_memory_sess(new_sess=False):
    """
	Creates tf.Session which dynamically takes GPU memory. Note that taken memory won't
	be freed.

	Parameters
	----------
	new_sess : bool
		If `new_sess == False`, the function will return previously created session.
		Set `new_sess` to True if you need a new one.

	Returns
	-------
	tf.Session
	"""
    global LOW_MEMORY_SESS
    if LOW_MEMORY_SESS is None or new_sess:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        LOW_MEMORY_SESS = tf.Session(config=config)
    return LOW_MEMORY_SESS


def get_fraction_memory_sess(fraction=0.5, new_sess=False):
    """
	Creates tf.Session which takes only a given fraction of a GPU memory.

	Parameters
	----------
	fraction : float
		How much of a GPU memory the session will take.
		Example: 0.5 - 50%, 0.7 - 70% of the memory.
	new_sess : bool
		If `new_sess == False`, the function will return previously created session.
		Set `new_sess` to True if you need a new one.

	Returns
	-------
	tf.Session
	"""
    global FRACTION_MEMORY_SESS
    if FRACTION_MEMORY_SESS is None or new_sess:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        FRACTION_MEMORY_SESS = tf.Session(config=config)
    return FRACTION_MEMORY_SESS


def freeze_model(checkpoint_path, protobuf_name, output_node_names):
    """
	Freezes model by converting its weights to constants and saves
	it to protobuf (.pb) file.

	Parameters
	----------
	checkpoint_path : str
		Path to the model's checkpoint file.
		There must be `.meta` file with the graph info.
		Example: 'model_folder/weights.ckpt'
	protobuf_name : str
		Name of the protobuf file which model will be
		saved to.
	output_node_names : list
		List of strings. Contains names of the output nodes
		of your model.
	"""
    sess = tf.Session()
    saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
    saver.restore(sess, checkpoint_path)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_node_names=output_node_names
    )

    with tf.gfile.GFile(protobuf_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
        print(f'Serialized to {protobuf_name}.')


def load_frozen_graph(protobuf_name):
    """
    Loads frozen graph from protobuf file.
    Loaded graph will become the default graph.

    Parameters
    ----------
    protobuf_name : str
        Name of the protobuf file to load from.

    Returns
    -------
    tf.Graph
    list
        List of Tensors or Operations in the graph.
    """
    with tf.gfile.GFile(protobuf_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    tensors_ops = tf.import_graph_def(graph_def, name='')
    return tf.get_default_graph(), tensors_ops
