import torch
from .model import create_model
from .trainer import ModelTrainer
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Creating model...")
    model = create_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.load("artifacts/data_loaders/train_loader.pth")
    val_loader = torch.load("artifacts/data_loaders/val_loader.pth")
    logger.info("Data Loaders loaded successfully.")

    logger.info("Starting training process..")
    trainer = ModelTrainer(model, device, train_loader, val_loader)
    trained_model, val_accuracies = trainer.train()

    torch.save(trained_model.state_dict(), 'artifacts.models/trained_model.pth')
    logger.info("Training completed. Model saved as 'trained_model.pth'")
    logger.info(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")



if __name__ == "__main__":
    main()