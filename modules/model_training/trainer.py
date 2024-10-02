#from logging import Logger
import torch
import torch.nn as nn
import torch.optim as optim
import time



class ModelTrainer:
    def __init__(self, model: nn.Module, device, train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, num_epochs: int = 3, 
                 learning_rate: float = 0.0001):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.val_accuracies = []

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (images, labels) in enumerate(self.train_loader):
            batch_start_time = time.time()
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{self.num_epochs}], "
                      f"Batch [{i+1}/{len(self.train_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Batch Time: {time.time() - batch_start_time:.2f}s")
                
        epoch_loss = running_loss / len(self.train_loader)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch [{epoch+1}/{self.num_epochs}] completed, "
              f"Average Loss: {epoch_loss:.4f}, "
              f"Epoch Time: {epoch_time:.2f}s")
        
        return epoch_loss

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                val_loss += self.criterion(outputs.squeeze(), labels).item()
                predicted = torch.round(torch.sigmoid(outputs.squeeze()))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (i + 1) % 10 == 0:  # Print every 10 batches
                    print(f"Validation Batch [{i+1}/{len(self.val_loader)}] processed")
        
        accuracy = 100 * correct / total
        self.val_accuracies.append(accuracy)
        return val_loss / len(self.val_loader), accuracy

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate()

            print(f"Epoch [{epoch+1}/{self.num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.2f}%")
        
        return self.model, self.val_accuracies