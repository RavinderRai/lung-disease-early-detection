import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_kaggle_dataset(dataset_identifier: str, download_path: str) -> None:
    """
    Downloads a Kaggle dataset to the specified directory.

    Args:
        dataset_identifier (str): The identifier of the dataset to download.
        download_path (str): The path to the directory where the dataset should be downloaded.

    Returns:
        None
    """
    load_dotenv(find_dotenv())

    # Set Kaggle API credentials from environment variables
    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

    # Authenticate using Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Make sure the download path exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)
        logging.info(f"Created download directory at: {download_path}")

    # Download the dataset
    logging.info(f"Downloading dataset: {dataset_identifier}")
    api.dataset_download_files(dataset_identifier, path=download_path, unzip=True)
    logging.info(f"Dataset downloaded and extracted to: {download_path}")

if __name__ == "__main__":
    # Kaggle dataset identifier from https://www.kaggle.com/datasets/nih-chest-xrays/data
    dataset_identifier = 'nih-chest-xrays/data'

    # Path to the custom directory on your storage device, modify this for your case if needed
    download_path = 'D:\\BigData'
    download_kaggle_dataset(dataset_identifier, download_path)
