import json
import re
from collections import Counter

import numpy as np
from keras.models import model_from_json

import utils


class CharacterCodec(object):
    def __init__(self, alphabet, maxlen):
        self.alphabet = list(sorted(set(alphabet)))
        self.index_alphabet = dict((c, i) for i, c in enumerate(self.alphabet))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.alphabet)))
        for i, c in enumerate(C[:maxlen]):
            X[i, self.index_alphabet[c]] = 1
        return X

    def try_encode(self, C, maxlen=None):
        try:
            return self.encode(C, maxlen)
        except KeyError:
            return None

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.alphabet[x] for x in X)


class Model(object):
    def __init__(self, config_file, model_file, weights_file):
        with open(config_file) as f:
            self.config = json.load(f)

        self.maxlen = self.config.get('MAXLEN', 32)
        self.invert = self.config.get('INVERT', True)
        self.ngram = self.config.get('NGRAM', 5)
        self.pad_words_input = self.config.get('PAD_WORDS_INPUT', True)

        self.codec = CharacterCodec(utils.ALPHABET, self.maxlen)
        if self.config.get('BASE_CODEC_INPUT', False):
            self.input_codec = CharacterCodec(utils.BASE_ALPHABET, self.maxlen)
        else:
            self.input_codec = self.codec

        with utils.timing('Create model'):
            with open(model_file) as f:
                self.model = model_from_json(f.read())
        with utils.timing('Compile model'):
            self.model.compile(loss='categorical_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])
        with utils.timing('Load weights'):
            self.model.load_weights(weights_file)

    def guess(self, words):
        text = ' '.join(words)
        text = utils.pad(text, self.maxlen)
        if self.invert:
            text = text[::-1]
        input_vec = self.input_codec.encode(text)
        preds = self.model.predict_classes(np.array([input_vec]), verbose=0)
        return self.codec.decode(preds[0], calc_argmax=False).strip('\x00')

    def add_accent(self, text):
        # lowercase the input text as we train the model on lowercase text only
        # but we keep the map of uppercase characters to restore cases in output
        is_uppercase_map = [c.isupper() for c in text]
        text = utils.remove_accent(text.lower())

        # for each block of words or symbols in input text, either append the symbols or
        # add accents for words and append them.
        outputs = []
        words_or_symbols_list = re.findall('\w[\w ]*|\W+', text)
        print('input:', words_or_symbols_list)
        for words_or_symbols in words_or_symbols_list:
            if utils.is_words(words_or_symbols):
                outputs.append(self._add_accent(words_or_symbols))
            else:
                outputs.append(words_or_symbols)
        print('output:', outputs)
        output_text = ''.join(outputs)

        # restore uppercase characters
        output_text = ''.join(c.upper() if is_upper else c
                              for c, is_upper in zip(output_text, is_uppercase_map))
        return output_text

    def _add_accent(self, phrase):
        grams = list(utils.gen_ngram(phrase.lower(), n=self.ngram, pad_words=self.pad_words_input))
        guessed_grams = list(self.guess(gram) for gram in grams)
        candidates = [Counter() for _ in range(len(guessed_grams) + self.ngram - 1)]
        for idx, gram in enumerate(guessed_grams):
            for wid, word in enumerate(re.split(' +', gram)):
                candidates[idx + wid].update([word])
        output = ' '.join(c.most_common(1)[0][0] for c in candidates if c)
        return output.strip('\x00 ')


def load_model(path):
    print('Loading model from: {}'.format(path))
    model_file = '{}/model.json'.format(path)
    config_file = '{}/config.json'.format(path)
    weights_file = '{}/weights.h5'.format(path)
    return Model(config_file, model_file, weights_file)
