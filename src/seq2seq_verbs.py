from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from pathlib import Path
from src.verb_processor import VerbProcessor


class Seq2SeqVerbs:
    def __init__(self, units, text_file, headers, rnn_type='vanilla', name='vanilla'):
        """
        Sequence to sequence model for verb generation
        :param units: number of rnn units
        :param text_file: training text file. should include target and training data
        :param headers: dict of target and training data
        :param rnn_type: choose rnn variants. vanilla, gru and lstm are available
        :param name: class name
        """
        self.infer_decoder_model = None
        self.model = None
        self.infer_encoder_model = None
        self.num_units = units
        self.rnn_type = rnn_type

        self.verb_processor = VerbProcessor(text_file, headers)

        self.name = name
        self.model_path = Path(__file__).parent.parent / 'models' / name

        if self.model_path.is_dir():
            self.load_model(self.model_path.resolve())
        else:
            self.create_model()

        self.create_inference_model()

    def create_model(self):
        num_token = self.verb_processor.num_token

        encoder_inputs = keras.Input(shape=(None, num_token), name='input')

        if self.rnn_type == 'gru':
            encoder = keras.layers.GRU(units=self.num_units, return_state=True, name='encoder')
        elif self.rnn_type == 'lstm':
            encoder = keras.layers.LSTM(units=self.num_units, return_state=True, name='encoder')
        else:
            encoder = keras.layers.SimpleRNN(units=self.num_units, return_state=True, name='encoder')

        if self.rnn_type == 'lstm':
            _, enc_h, enc_c = encoder(encoder_inputs)
            encoder_states = [enc_h, enc_c]
        else:
            _, encoder_states = encoder(encoder_inputs)

        decoder_inputs = keras.Input(shape=(None, num_token), name='dec_input')

        if self.rnn_type == 'gru':
            decoder = keras.layers.GRU(units=self.num_units, return_state=True, return_sequences=True, name='decoder')
        elif self.rnn_type == 'lstm':
            decoder = keras.layers.LSTM(units=self.num_units, return_state=True, return_sequences=True, name='decoder')
        else:
            decoder = keras.layers.SimpleRNN(units=self.num_units, return_state=True, return_sequences=True, name='decoder')

        if self.rnn_type == 'lstm':
            decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
        else:
            decoder_outputs, _ = decoder(decoder_inputs, initial_state=encoder_states)

        decoder_dense = keras.layers.Dense(num_token, activation='softmax', name='dec_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def load_model(self, model_file, evaluate=False):
        self.model = keras.models.load_model(model_file)

        if evaluate:
            self.evaluate()

    def evaluate(self):
        x_train = self.verb_processor.one_hot_train
        y_train = self.verb_processor.one_hot_target
        y_ahead_train = self.verb_processor.one_hot_ahead

        self.model.evaluate(
            [x_train, y_train],
            y_ahead_train)

    def create_inference_model(self):
        # Inference Model
        encoder_inputs = self.model.input[0]

        if self.rnn_type == 'lstm':
            _, enc_h, enc_d = self.model.layers[2].output
            encoder_states = [enc_h, enc_d]
        else:
            _, encoder_states = self.model.layers[2].output

        self.infer_encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = self.model.input[1]

        decoder = self.model.layers[3]

        if self.rnn_type == 'lstm':
            dec_state_input_h = keras.Input(shape=(self.num_units,))
            dec_state_input_c = keras.Input(shape=(self.num_units,))

            decoder_states_inputs = [dec_state_input_h, dec_state_input_c]

            decoder_outputs, dec_h, dec_d = decoder(
                decoder_inputs, initial_state=decoder_states_inputs
            )

            decoder_states = [dec_h, dec_d]
        else:
            decoder_states_inputs = keras.Input(shape=(self.num_units,))

            decoder_outputs, decoder_states = decoder(
                decoder_inputs, initial_state=decoder_states_inputs
            )

        decoder_dense = self.model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)

        if self.rnn_type == 'lstm':
            self.infer_decoder_model = keras.Model(
                [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
            )
        else:
            self.infer_decoder_model = keras.Model(
                [decoder_inputs, decoder_states_inputs], [decoder_outputs, decoder_states]
            )

    def train(self, batch_size=64, epochs=800, verbose=0):

        x_train = self.verb_processor.one_hot_train
        y_train = self.verb_processor.one_hot_target
        y_ahead_train = self.verb_processor.one_hot_ahead

        self.model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        cb = EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)

        self.history = self.model.fit(
            [x_train, y_train],
            y_ahead_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=[cb]
        )

        self.model.save(self.model_path.resolve())

    def infer(self, text: str):
        one_hot_seq = self.verb_processor.get_one_hot_sample(text_sample=text)
        states_value = self.infer_encoder_model.predict(one_hot_seq, verbose=0)

        target_seq = np.zeros((1, 1, self.verb_processor.num_token))
        target_seq[0, 0, self.verb_processor.start_index] = 1.0

        is_finished = False
        generated_idx = []

        while not is_finished:
            if self.rnn_type == 'lstm':
                dec_out, states_value_h, states_value_d = self.infer_decoder_model.predict(
                    [target_seq] + states_value,
                    verbose=0)
                states_value = [states_value_h, states_value_d]
            else:
                dec_out, states_value = self.infer_decoder_model.predict(
                    [target_seq, states_value],
                    verbose=0)
            out_idx = np.argmax(dec_out)

            if out_idx == self.verb_processor.stop_index or len(generated_idx) > self.verb_processor.max_pad:
                is_finished = True
            else:
                generated_idx.append(out_idx)

            target_seq = np.zeros((1, 1, self.verb_processor.num_token))
            target_seq[0, 0, out_idx] = 1.0

        res = self.verb_processor.tokenizer.sequences_to_texts([generated_idx])

        return ''.join(res).replace(' ', '')

    def infer_list(self, text_list):
        infer_list = list()
        for i in range(len(text_list)):
            inferred = self.infer(text_list[i])
            infer_list.append(inferred)
        return infer_list


if __name__ == '__main__':

    lstm_s2s = Seq2SeqVerbs(
        units=16,
        rnn_type='lstm',
        text_file='../data/irr_verb_list.csv',
        headers={'train': 'infinitive', 'target': 'past simple'},
        name='lstm16')
    #lstm_s2s.train()

    print(lstm_s2s.infer('fight'))
