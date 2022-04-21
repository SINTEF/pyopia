import urllib.request
import zipfile
import os


def get_example_silc_image():
    filename = 'D20181101T142731.838206.silc'
    if os.path.exists(filename):
        return filename
    url = 'https://pysilcam.blob.core.windows.net/test-data/D20181101T142731.838206.silc'
    urllib.request.urlretrieve(url, filename)
    return filename


def get_example_model():
    model_filename = 'keras_model.h5'
    if os.path.exists(model_filename):
        return model_filename
    # -- Download and unzip the model --#
    url = 'https://github.com/SINTEF/PySilCam/wiki/ml_models/keras_model.zip'
    model_filename = 'keras_model.zip'
    urllib.request.urlretrieve(url, model_filename)
    with zipfile.ZipFile(model_filename, 'r') as zipit:
        zipit.extractall()
    return model_filename
