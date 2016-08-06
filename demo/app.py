""" Vietnamese Accent Predictor demo """
import io
import os

import hug

import model

MODEL_DIR = os.environ.get('MODELS_DIR', 'models')
MODEL_VERSION = os.environ.get('MODEL_VERSION', 'v1')
the_model = model.load_model(os.path.join(MODEL_DIR, MODEL_VERSION))

with io.open('static/index.html', encoding='utf-8') as f:
    index_page = f.read()


@hug.request_middleware()
def enable_cors_all(request, response):
    response.set_header('Access-Control-Allow-Origin', '*')


@hug.get('/', output=hug.output_format.html)
def index():
    """ Demo homepage """
    return index_page


@hug.get(examples='text=co%20gai%20den%20tu%20hom%20qua')
def accented(text: hug.types.text, hug_timer=3):
    """ Add accent to given plain text """
    return {
        'original': text,
        'with_accent': the_model.add_accent(text),
        'took': float(hug_timer)
    }


if __name__ == '__main__':
    accented.interface.cli()
