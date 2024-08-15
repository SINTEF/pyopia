import urllib.request
import zipfile
import os
import gdown


def get_file_from_pysilcam_blob(filename, download_directory='./'):
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


def get_gas_from_pysilcam_blob(download_directory='./gas_silcam_images'):
    '''Downloads a specified filename from the pysilcam.blob into the working dir. if it doesn't already exist

    only works for known filenames that are on this blob

    Parameters
    ----------
    filename : string
        known filename on the blob

    '''
    if os.path.exists(os.path.join(download_directory)):
        print(download_directory, 'already exists. Returning nothing')
        return
    os.makedirs(download_directory, exist_ok=False)
    url = 'https://pysilcam.blob.core.windows.net/test-data/gas.zip'
    print('Downloading....')
    urllib.request.urlretrieve(url, download_directory + '/gas.zip')
    print('Unzipping....')
    with zipfile.ZipFile(os.path.join(download_directory, 'gas.zip'), 'r') as zipit:
        zipit.extractall(download_directory)
    print('Done.')


def get_oil_from_pysilcam_blob(download_directory='./oil_silcam_images'):
    '''Downloads a specified filename from the pysilcam.blob into the working dir. if it doesn't already exist

    only works for known filenames that are on this blob

    Parameters
    ----------
    filename : string
        known filename on the blob

    '''
    local_zip_file = os.path.join(download_directory, 'oil.zip')
    extract_dir = os.path.join(download_directory, 'oil')
    if os.path.exists(local_zip_file):
        print(download_directory, 'already exists. Returning nothing')
        return extract_dir
    os.makedirs(download_directory, exist_ok=False)
    url = 'https://pysilcam.blob.core.windows.net/test-data/oil.zip'
    print('Downloading....')
    urllib.request.urlretrieve(url, local_zip_file)
    print('Unzipping....')
    with zipfile.ZipFile(local_zip_file, 'r') as zipit:
        zipit.extractall(download_directory)
    print('Done.')

    return extract_dir


def get_example_silc_image(download_directory='./'):
    '''calls `get_file_from_pysilcam_blob` for a silcam iamge

    Returns
    -------
    string
        filename
    '''
    filename = 'D20181101T142731.838206.silc'
    if os.path.isfile(filename):
        print('Example image already exists. Skipping download.')
        return filename
    get_file_from_pysilcam_blob(filename, download_directory)
    return filename


def get_example_model(download_directory='./'):
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


def get_example_hologram_and_background(download_directory='./'):
    '''calls `get_file_from_pysilcam_blob` for a raw hologram, and its associated background image.

    Returns
    -------
    string
        holo_filename

    string
        holo_background_filename
    '''
    holo_filename = '001-2082.pgm'
    holo_background_filename = 'imbg-' + holo_filename
    get_file_from_pysilcam_blob(holo_filename, download_directory)
    get_file_from_pysilcam_blob(holo_background_filename, download_directory)

    holo_filename = os.path.join(download_directory, holo_filename)
    holo_background_filename = os.path.join(download_directory, holo_background_filename)

    return holo_filename, holo_background_filename


def get_folder_from_holo_repository(foldername="holo_test_data_01", existsok=False):
    '''Downloads a specified folder from the holo testing repository into the working dir. if it doesn't already exist

    only works for known folders that are on the GoogleDrive repository
    by default will download a known-good folder. Additional elif statements can be added to implement additional folders.

    Parameters
    ----------
    foldername : string
        known filename on the blob
    existsok : (bool, optional)
        if True, then don't download if the specified folder already exists, defaults to False

    '''
    if foldername == "holo_test_data_01":
        url = 'https://drive.google.com/drive/folders/1yNatOaKdWwYQp-5WVEDItoibr-k0lGsP?usp=share_link'

    elif foldername == "holo_test_data_02":
        url = "https://drive.google.com/drive/folders/1E5iNSyfeKcVMLVe4PNEwF2Q2mo3WVjF5?usp=share_link"

    else:
        foldername == "holo_test_data_01"
        url = 'https://drive.google.com/drive/folders/1yNatOaKdWwYQp-5WVEDItoibr-k0lGsP?usp=share_link'

    if os.path.exists(foldername) and existsok:
        print(foldername + ' already exists. Skipping download.')
        return foldername

    gdown.download_folder(url, quiet=True, use_cookies=False)
    return foldername
