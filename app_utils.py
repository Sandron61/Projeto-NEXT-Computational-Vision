# app_utils.py
import traceback
import logging
import os
from werkzeug.utils import secure_filename

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pt'}

def secure_filename_custom(filename):
    return secure_filename(filename)

class TryExcept:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Ocorreu um erro na função '{self.func.__name__}': {e}")
            traceback.print_exc()
            return None

