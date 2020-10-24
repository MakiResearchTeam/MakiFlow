from .graph_entities import MakiTensor


def to_makitensor(tf_tensor, name, parent_layer=None, parent_tensor_names=None, previous_tensors=None):
    """
    Converts a TensorFlow tensor to MakiTensor.
    Parameters
    ----------
    tf_tensor : tf.Tensor
        Actual data tensor.
    name : str
        Name of the tensor.
    parent_layer : MakiLayer
        Layer that produced this tensor. If it is set to None, it may cause an error during architecture
        save of the model.
    parent_tensor_names : list
        List of names of the MakiTensors that were used during creation of this tensor.
    previous_tensors : dict
        Dictionary containing all the tensors appearing earlier in the computational graph.

    Returns
    -------
    MakiTensor
    """
    if parent_tensor_names is None:
        parent_tensor_names = []
    if previous_tensors is None:
        previous_tensors = {}

    return MakiTensor(
        data_tensor=tf_tensor,
        parent_layer=parent_layer,
        parent_tensor_names=parent_tensor_names,
        previous_tensors=previous_tensors
    )


def pack_data(feed_dict_config, data: list):
    """
    Packs data into a dictionary with pairs (tf.Tensor, data).
    This dictionary is then used as the `feed_dict` argument in the session.run() method.
    Parameters
    ----------
    feed_dict_config : dict
        Contains pairs (MakiTensor, int) or (tf.Tensor, int), where int is the index of the data point in the `data`.
    data : list
        The data to pack.

    Returns
    -------
    dict
        Dictionary with packed data.
    """

    feed_dict = dict()
    for t, i in feed_dict_config.items():
        # If the `t` is a tf.Tensor
        data_tensor = t
        if isinstance(t, MakiTensor):
            data_tensor = t.get_data_tensor()
        feed_dict[data_tensor] = data[i]
    return feed_dict
