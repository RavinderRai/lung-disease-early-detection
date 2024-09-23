import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_kaggle_dataset(dataset_identifier, download_path):
    # Load environment variables from the .env file
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
    # Kaggle dataset identifier from the provided URL
    dataset_identifier = 'nih-chest-xrays/data'

    # Path to the custom directory on your storage device
    download_path = 'D:\\BigData'

    # Call the function to download the dataset
    download_kaggle_dataset(dataset_identifier, download_path)
