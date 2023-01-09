'''
This is a submodule for providing convienient access to online testdata
for use in the tests contain within this submodule.
'''

import urllib.request
import zipfile
import os


def get_file_from_pysilcam_blob(filename, download_directory):
    '''Downloads a specified filename from the pysilcam.blob into the working dir. if it doesn't already exist

    only works for known filenames that are on this blob

    Parameters
    ----------
    filename : string
        known filename on the blob

    '''
    if os.path.exists(os.path.join(download_directory, filename)):
        return filename
    url = 'https://pysilcam.blob.core.windows.net/test-data/' + filename
    urllib.request.urlretrieve(url, os.path.join(download_directory, filename))


def get_example_silc_image(download_directory):
    '''calls `get_file_from_pysilcam_blob` for a silcam iamge

    Returns
    -------
    string
        filename
    '''
    filename = 'D20181101T142731.838206.silc'
    get_file_from_pysilcam_blob(filename, download_directory)
    return filename


def get_example_model(download_directory):
    '''Downloads and unzips an example trained CNN model from the pysilcam.blob
    into the working dir. if it doesn't already exist.

    Returns
    -------
    string
        model_filename
    '''
    model_filename = 'keras_model.h5'
    if os.path.exists(model_filename):
        return model_filename
    # -- Download and unzip the model --#
    url = 'https://github.com/SINTEF/PySilCam/wiki/ml_models/keras_model.zip'
    model_filename = 'keras_model.zip'
    urllib.request.urlretrieve(url, os.path.join(download_directory, model_filename))
    with zipfile.ZipFile(os.path.join(download_directory, model_filename), 'r') as zipit:
        zipit.extractall(download_directory)
    model_filename = 'keras_model.h5'
    return os.path.join(download_directory, model_filename)
