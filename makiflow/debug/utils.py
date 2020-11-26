

# debug_message
def d_msg(context, msg_content):
    """
    A simple utility that adds the context to the message string.

    Parameters
    ----------
    context : str
        A string to be appended.
    msg_content: str
        The content of the message.
    Returns
    -------
        Modified message.
    """
    return f'{context} / Message = {msg_content}'



