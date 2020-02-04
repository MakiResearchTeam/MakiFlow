EPOCH = 'Epoch: '


def print_train_info(epoch, *args):
    output = ''
    for value_name, value in args:
        value_name = value_name.lower() + ': '
        value_name[0] = value_name[0].upper()
        output += value_name + '{:0.4f}'.format(value) + ' '

    print(EPOCH, epoch, output)


def moving_average(old_val, new_val, iteration):
    if iteration == 0:
        return new_val
    else:
        return old_val * 0.9 + new_val * 0.1


def new_optimizer_used():
    print('New optimizer is used.')


def loss_is_built():
    print('Loss is built.')
