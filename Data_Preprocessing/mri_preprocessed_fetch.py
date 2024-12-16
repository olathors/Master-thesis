import os

def get_preprocessed_imagenames(path):

    for subdir, dirs, files in os.walk(path):
        return files