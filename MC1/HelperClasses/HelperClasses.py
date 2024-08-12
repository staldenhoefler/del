import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import wandb
import time

class HelperClass:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def train(self, train_loader,test_loader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            start_traintime = time.time()
            for i, (data) in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            train_loss = running_loss / (i+1)
            endtime_train = time.time() - start_traintime
            self.evaluate(test_loader=test_loader, train_loader=train_loader, train_loss=train_loss, endtime_train=endtime_train)
            print(f"Epoch {epoch+1}, Loss: {train_loss}")
        print("Finished Training")
        wandb.finish()
        return self.model


    def evaluate(self, test_loader, train_loader, train_loss, endtime_train):
        # Evaluate the model on test_loader
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        starttime_test = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        test_loss = test_loss / total

        # Evaluate the model on the train_loader
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        endtime_test = time.time() - starttime_test
        wandb.log(
            {
                "test_accuracy": test_accuracy,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "time_train": endtime_train,
                "time_test": endtime_test,

            }
        )




def prepare_data(batch_size):
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    return train_dataloader, test_loader

def wandb_login(dict):

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="del-MC1",

        # track hyperparameters and run metadata
        config=dict
    )
