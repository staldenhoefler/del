import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision import models
import wandb
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class HelperClass:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.wandb_run = None

    def train(self, train_loader, test_loader, num_epochs, lambda_l2=0):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            start_traintime = time.time()
            self.model.train()
            for i, (data) in enumerate(train_loader, 0):
                inputs, labels = data

                # for k in range(len(inputs)):
                #   plt.imshow(inputs[k].permute(1, 2, 0))
                #  plt.show()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                l2_penalty = 0.0

                for p in self.model.parameters():
                    l2_penalty += p.pow(2).sum()

                loss = loss + (lambda_l2 * l2_penalty)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            train_loss = running_loss / (i + 1)
            endtime_train = time.time() - start_traintime
            test_accuracy, train_accuracy, train_loss, test_loss = self.evaluate(test_loader=test_loader,
                                                                                 train_loader=train_loader,
                                                                                 train_loss=train_loss,
                                                                                 endtime_train=endtime_train)
            print(
                f"Epoch {epoch + 1}, Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
        print("Finished Training")
        wandb.finish()
        return self.model

    def evaluate(self, test_loader, train_loader, train_loss, endtime_train):
        # Evaluate the model on test_loader
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        batches = 0
        test_loss = 0
        starttime_test = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                batches += 1
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        test_loss = test_loss / batches

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
        return test_accuracy, train_accuracy, train_loss, test_loss


def prepare_data(batch_size):
    transform = transforms.Compose([
        #transforms.RandomChoice([
        #    transforms.RandomHorizontalFlip()
        #    , transforms.RandomRotation(45)
        #    , transforms.RandomAutocontrast()
        #    , transforms.RandomVerticalFlip()
        #    , transforms.RandomResizedCrop(32, scale=(0.5, 1.0))
        #])
           transforms.ToTensor()
        #,   transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=True)
        , transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784), inplace=True)
    ])

    training_data = datasets.CIFAR10(
        root="/mnt/nas05/clusterdata01/home2/d.schatzmann/data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root="/mnt/nas05/clusterdata01/home2/d.schatzmann/data",
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             persistent_workers=True, prefetch_factor=2)
    return train_dataloader, test_loader


def wandb_login(dict, name=None):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="del-MC1",
        name=name,
        # track hyperparameters and run metadata
        config=dict
    )


def get_run_hist(run_id):
    api = wandb.Api()
    run = api.run(path=f'del-MC1/{run_id}')
    history = run.history()
    return history


def test_model(name: str, linear_layers: int, conv_layers: int, kernel_size=5, stride: int = 1, padding: int = 0):
    batch_sizes = [32]
    learning_rates = [0.01, 0.1]
    lambda_regs = [0.0]
    epochs = 25

    for batch_size in batch_sizes:
        for lr in learning_rates:
            for lambda_reg in lambda_regs:
                sgdModel = SGDModel()
                # Loss function
                criterion = nn.CrossEntropyLoss()

                # Optimizer
                optimizer = optim.SGD(sgdModel.parameters(), lr=lr)

                model_class = HelperClass(sgdModel, criterion, optimizer)

                train_loader, test_loader = prepare_data(batch_size)

                dict = {
                    "dataset": "CIFAR-10-Normalized",
                    "epochs": epochs,
                    "linear_layers": linear_layers,
                    "learning_rate": lr,
                    "architecture": "CNN",
                    "batch_size": batch_size,
                    "conv_layers": conv_layers,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "padding": padding,
                    "Lambda": lambda_reg,
                    "dropout": 0.0,
                }

                wandb_login(dict, name=f'CNN-{name}-bs{batch_size}-lr{lr}')

                trained_model = model_class.train(train_loader, test_loader, epochs, lambda_l2=lambda_reg)


class SGDModel(nn.Module):
    def __init__(self):
        super(SGDModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # 30*30*32
        self.conv2 = nn.Conv2d(32, 128, 3)  # 28*28*128
        self.conv3 = nn.Conv2d(128, 256, 3)  # 26*26*256
        self.conv4 = nn.Conv2d(256, 512, 3) # 24*24*512
        self.pool1 = nn.MaxPool2d(2, 2)  # 12*12*512
        self.fc1 = nn.Linear(73728, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

trained_model = test_model("SGD", 4, 5, kernel_size=3, stride=1, padding=0)
