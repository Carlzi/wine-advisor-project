import csv
import os
from google.cloud import storage
import pickle

def load_csv(csv_file):
    resources = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            resources.append(row[0])
    return resources

def load_csv_gcp(filename:str, filepath:str, saving_path:str) -> None:
    """Charge un fichier

    Args:
        repo (str): _description_
    """
    root_dir = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(root_dir, saving_path)):
        os.makedirs(os.path.join(root_dir, 'tmp', saving_path), exist_ok=True)
    local_path = os.path.join(root_dir, 'tmp', saving_path, filename)

    if not os.path.exists(local_path):
        print(f'Loading file {filename} from GCS')
        client = storage.Client()
        bucket = client.bucket(os.getenv('BUCKET_NAME'))
        blob = bucket.blob(f"{filepath}/{filename}")

        blob.download_to_filename(local_path)
    return load_csv(local_path)


def load_model_gcp(filename:str, filepath:str, saving_path:str) -> None:
    """Charge un fichier

    Args:
        repo (str): _description_
    """
    root_dir = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(root_dir, saving_path)):
        os.makedirs(os.path.join(root_dir, 'tmp', saving_path), exist_ok=True)
    local_path = os.path.join(root_dir, 'tmp', saving_path, filename)

    if not os.path.exists(local_path):
        print(f'Loading file {filename} from GCS')
        client = storage.Client()
        bucket = client.bucket(os.getenv('BUCKET_NAME'))
        blob = bucket.blob(f"{filepath}/{filename}")

        blob.download_to_filename(local_path)
    return pickle.load(open(local_path, 'rb'))
