from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pandas as pd


class VerbProcessor:
    def __init__(self, text_file, headers, prefix='\t', postfix='\n'):
        """
        class for text processing
        :param text_file: raining text file. should include target and training data
        :param headers: ict of target and training data
        :param prefix: character denoting the start of a word
        :param postfix: character denoting the end of a word
        """
        self.prefix = prefix
        self.postfix = postfix

        self.dataset_raw = pd.read_csv(text_file)
        self.dataset = self.dataset_raw.apply(lambda x: prefix + x + postfix)

        self.num_data = len(self.dataset)

        self.train = self.dataset[headers['train']].to_list()
        self.target = self.dataset[headers['target']].to_list()

        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(self.train + self.target)
        self.num_classes = len(self.tokenizer.word_index)+1

        self.train_seq = self.tokenizer.texts_to_sequences(self.train)
        self.target_seq = self.tokenizer.texts_to_sequences(self.target)
        self.ahead_seq = get_ahead_target(self.target_seq)

        seq = (self.train_seq + self.target_seq + self.ahead_seq)

        self.max_pad = find_max_list(seq)

        padded = pad_sequences(seq, maxlen=self.max_pad, padding='post')

        one_hot = to_categorical(padded, num_classes=self.num_classes)

        self.one_hot_train = one_hot[:self.num_data]
        self.one_hot_target = one_hot[self.num_data:2*self.num_data]
        self.one_hot_ahead = one_hot[2*self.num_data:]

        self.input_dim = self.one_hot_train.shape[1]
        self.output_dim = self.one_hot_target.shape[1]
        self.num_token = self.one_hot_train.shape[-1]

        self.start_index = self.tokenizer.texts_to_sequences([self.prefix])[0][0]
        self.stop_index = self.tokenizer.texts_to_sequences([self.postfix])[0][0]

    def get_one_hot_sample(self, text_sample):
        text_seq = self.prefix + text_sample + self.postfix
        text_seq = self.tokenizer.texts_to_sequences([text_seq])
        text_seq = pad_sequences(text_seq, maxlen=self.max_pad, padding='post')
        text_seq = to_categorical(text_seq, num_classes=self.num_classes)
        return text_seq

    def get_one_hot(self, text_sample: list()):
        samples_seq = pd.Series(text_sample).apply(lambda x: self.prefix + x + self.postfix)

        samples_seq = self.tokenizer.texts_to_sequences(samples_seq.to_list())

        samples_seq = pad_sequences(samples_seq, maxlen=self.max_pad, padding='post')
        samples_seq = to_categorical(samples_seq, num_classes=self.num_classes)

        return samples_seq


def get_ahead_target(seq: list(), stop_val=0):
    res = list()
    for i in range(len(seq)):
        temp = list()
        for j in range(1, len(seq[i])):
            temp.append(seq[i][j])
        temp.append(stop_val)
        res.append(temp)
    return res


def find_max_list(lst: list()):
    list_len = [len(i) for i in lst]
    return max(list_len)
