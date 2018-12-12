import requests
import zipfile
import os


def download_file_from_google_drive(id, destination):
    """
    Using from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """

    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_and_remove_zip(directory_path, zip_file):

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_path)

    os.remove(zip_file)
