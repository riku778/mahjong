import tensorflow as tf
from keras.api._v2 import keras
from keras.layers import Input, Dense
from keras.models import Model

import params


class DNNModel:
    def __init__(
            self,
            input_size: int = params.INPUT_SIZE,
            hidden1_size: int = params.HIDDEN1_SIZE,
            hidden2_size: int = params.HIDDEN2_SIZE,
            hidden3_size: int = params.HIDDEN3_SIZE,
            dropout_rate_inp: float = params.DEROPOUT_RATE_INP,
            dropout_rate_hid: float = params.DEROPOUT_RATE_HID,
            output_size: int = params.OUTPUT_SIZE) -> None:
        self.input: Input = Input(shape=(input_size,), name='input')
        self.hidden1: Dense = Dense(hidden1_size, activation='relu', name='hidden1')
        self.hidden2: Dense = Dense(hidden2_size, activation='relu', name='hidden2')
        self.hidden3: Dense = Dense(hidden3_size, activation='relu', name='hidden3')
        self.dropout_inp = keras.layers.Dropout(dropout_rate_inp)
        self.dropout_hid = keras.layers.Dropout(dropout_rate_hid)
        self.output: Dense = Dense(output_size, name='output')

    def build(self) -> Model:
        input = self.input
        x = self.hidden1(input)
        x = self.dropout_inp(x)
        x = self.hidden2(x)
        x = self.dropout_hid(x)
        x = self.hidden3(x)
        output = self.output(x)
        return Model(inputs=input, outputs=output)