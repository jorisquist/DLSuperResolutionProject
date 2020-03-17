import requests
url_91 = 'https://drive.google.com/uc?export=download&id=1eVfd2Snh5bCl0ulMsRE4ker_p-o1M_lm'
url_set5 = 'https://drive.google.com/uc?export=download&id=1Cr4puJ1UpkXrGpzdpqZLNhZiZ2vaimoi'
url_set14 = 'https://drive.google.com/uc?export=download&id=1PQus6Glc3VsfVIywG6MAMBBBZVyyF_gB'
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

download_url(url_91, './91.zip')
download_url(url_set5, './set5.zip')
download_url(url_set14, './set14.zip')

from zipfile import ZipFile

with ZipFile('91.zip', 'r') as zipObj:
  zipObj.extractall('./train_data')

with ZipFile('set5.zip', 'r') as zipObj:
  zipObj.extractall('./test_data')

with ZipFile('set14.zip', 'r') as zipObj:
  zipObj.extractall('./test_data')