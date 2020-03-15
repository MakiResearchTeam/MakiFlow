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

EPOCH = 'Epoch:'


def print_train_info(epoch, *args):
    output = ''
    for value_name, value in args:
        value_name = value_name.lower() + ': '
        value_name = value_name.title()
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
