import re
import string
import time
from contextlib import contextmanager

ACCENTED_CHARS = {
    'a': u'a á à ả ã ạ â ấ ầ ẩ ẫ ậ ă ắ ằ ẳ ẵ ặ',
    'o': u'o ó ò ỏ õ ọ ô ố ồ ổ ỗ ộ ơ ớ ờ ở ỡ ợ',
    'e': u'e é è ẻ ẽ ẹ ê ế ề ể ễ ệ',
    'u': u'u ú ù ủ ũ ụ ư ứ ừ ử ữ ự',
    'i': u'i í ì ỉ ĩ ị',
    'y': u'y ý ỳ ỷ ỹ ỵ',
    'd': u'd đ',
}

ACCENTED_TO_BASE_CHAR_MAP = {}
for c, variants in ACCENTED_CHARS.items():
    for v in variants.split(' '):
        ACCENTED_TO_BASE_CHAR_MAP[v] = c

# \x00 is the padding characters
BASE_ALPHABET = set('\x00 _' + string.ascii_lowercase + string.digits)
ALPHABET = BASE_ALPHABET.union(set(''.join(ACCENTED_TO_BASE_CHAR_MAP.keys())))


def is_words(text):
    return re.fullmatch('\w[\w ]*', text)


def remove_accent(text):
    """ remove accent from text """
    return u''.join(ACCENTED_TO_BASE_CHAR_MAP.get(char, char) for char in text)


def pad(phrase, maxlen):
    """ right pad given string with \x00 to exact "maxlen" length """
    return phrase + u'\x00' * (maxlen - len(phrase))


def gen_ngram(words, n=3, pad_words=True):
    """ gen n-grams from given phrase or list of words """
    if isinstance(words, str):
        words = re.split('\s+', words.strip())

    if len(words) < n:
        if pad_words:
            words += ['\x00'] * (n - len(words))
        yield tuple(words)
    else:
        for i in range(len(words) - n + 1):
            yield tuple(words[i: i + n])


@contextmanager
def timing(label):
    begin = time.monotonic()
    print(label, end='', flush=True)
    try:
        yield
    finally:
        duration = time.monotonic() - begin
    print(': took {:.2f}s'.format(duration))
