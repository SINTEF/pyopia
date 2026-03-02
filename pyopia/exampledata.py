import urllib.request
import zipfile
import os
import gdown
from pathlib import Path

import logging

logger = logging.getLogger()


def get_classifier_database_from_pysilcam_blob(download_directory="./"):
    """Downloads and unzips the silcam_database of labelled example images from pysilcam.blob
    into the working dir. if it doesn't already exist

    Parameters
    ----------
    download_directory : string
        directory to download and unzip the silcam_database.zip into. Defaults to "./"
        
    Returns
    -------
    string
        download_directory, the directory that the silcam_database.zip was downloaded and unzipped into

    """
    if os.path.exists(os.path.join(download_directory)):
        logger.info(download_directory, "already exists. Returning nothing")
        return download_directory
    os.makedirs(download_directory, exist_ok=False)
    url = "https://pysilcam.blob.core.windows.net/test-data/silcam_database.zip"
    logger.info("Downloading....")
    urllib.request.urlretrieve(url, download_directory + "/silcam_database.zip")
    logger.info("Unzipping....")
    with zipfile.ZipFile(
        os.path.join(download_directory, "silcam_database.zip"), "r"
    ) as zipit:
        zipit.extractall(os.path.join(download_directory, "../"))
    logger.info("Removing zip file")
    os.remove(os.path.join(download_directory, "silcam_database.zip"))
    logger.info("Done.")
    return download_directory


def get_file_from_pysilcam_blob(filename, download_directory="./"):
    """Downloads a specified filename from the pysilcam.blob into the working dir. if it doesn't already exist

    only works for known filenames that are on this blob

    Parameters
    ----------
    filename : string
        known filename on the blob
    download_directory : string
        directory to download the file into. Defaults to "./"

    Returns
    -------
    string
        filename of the downloaded file

    """
    if os.path.exists(os.path.join(download_directory, filename)):
        return filename
    url = "https://pysilcam.blob.core.windows.net/test-data/" + filename
    urllib.request.urlretrieve(url, os.path.join(download_directory, filename))
    return download_directory


def get_example_silc_image(download_directory="./"):
    """calls `get_file_from_pysilcam_blob` for a silcam iamge

    Parameters
    ----------
    download_directory : string
        directory to download the file into. Defaults to "./"
        
    Returns
    -------
    string
        filename of the downloaded silcam image
    
    """
    filename = "D20181101T142731.838206.silc"
    if os.path.isfile(filename):
        logger.info("Example image already exists. Skipping download.")
        return filename
    get_file_from_pysilcam_blob(filename, download_directory)
    return filename


def get_example_model(download_directory="./"):
    """Download PyOPIA default CNN model classifier

    Download from the pysilcam blob storage into the working dir.
    If the file exists, skip the download.
    
    Parameters
    ----------
    download_directory : string
        directory to download the file into. Defaults to "./"

    Returns
    -------
    string
        model_filename
    """
    model_filename = "pyopia-default-classifier-20250409.keras"
    model_path = Path(download_directory, model_filename)
    model_url = (
        f"https://pysilcam.blob.core.windows.net/test-data/{str(model_filename)}"
    )
    if not model_path.exists():
        logger.info("Downloading example model...")
        urllib.request.urlretrieve(model_url, model_path)
    return str(model_path)


def get_example_hologram_and_background(download_directory="./"):
    """calls `get_file_from_pysilcam_blob` for a raw hologram, and its associated background image.

    Parameters
    ----------
    download_directory : string
        directory to download the file into. Defaults to "./"

    Returns
    -------
    string
        holo_filename

    string
        holo_background_filename
    """
    holo_filename = "001-2082.pgm"
    holo_background_filename = "imbg-" + holo_filename
    get_file_from_pysilcam_blob(holo_filename, download_directory)
    get_file_from_pysilcam_blob(holo_background_filename, download_directory)

    holo_filename = os.path.join(download_directory, holo_filename)
    holo_background_filename = os.path.join(
        download_directory, holo_background_filename
    )

    return holo_filename, holo_background_filename


def get_folder_from_holo_repository(foldername="holo_test_data_01", existsok=False):
    """Downloads a specified folder from the holo testing repository into the working dir. if it doesn't already exist

    only works for known folders that are on the GoogleDrive repository
    by default will download a known-good folder. Additional elif statements can be added to implement additional folders.

    Parameters
    ----------
    foldername : string
        known filename on the blob
    existsok : (bool, optional)
        if True, then don't download if the specified folder already exists, defaults to False

    """
    if foldername == "holo_test_data_01":
        url = "https://drive.google.com/drive/folders/1yNatOaKdWwYQp-5WVEDItoibr-k0lGsP?usp=share_link"

    elif foldername == "holo_test_data_02":
        url = "https://drive.google.com/drive/folders/1E5iNSyfeKcVMLVe4PNEwF2Q2mo3WVjF5?usp=share_link"

    else:
        foldername == "holo_test_data_01"
        url = "https://drive.google.com/drive/folders/1yNatOaKdWwYQp-5WVEDItoibr-k0lGsP?usp=share_link"

    if os.path.exists(foldername) and existsok:
        logger.info(foldername + " already exists. Skipping download.")
        return foldername

    gdown.download_folder(url, quiet=True, use_cookies=False)
    return foldername
