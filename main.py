import sys
import logging
from modules.data_processing.run_data_processing import main as run_data_processing
from modules.model_training.train_model import main as train_model
from modules.app.app import main as launch_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        logger.error("Please specify a module to run: 'data', 'train', or 'app'")
        return

    option = sys.argv[1]

    if option == 'data':
        logger.info("Running data processing...")
        run_data_processing("artifacts/labels_data/lung_disease_labels.csv")
    elif option == 'train':
        num_epochs_input = input("Enter the number of epochs (default is 3): ")
        num_epochs = int(num_epochs_input) if num_epochs_input else 3
        logger.info(f"Training model for {num_epochs} epochs...")
        train_model(num_epochs)
    elif option == 'app':
        logger.info("Launching application...")
        launch_app()
    else:
        logger.error("Invalid option. Please choose 'data', 'train', or 'app'.")

if __name__ == "__main__":
    main()