import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from pyspark.sql.dataframe import DataFrame
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os

class CNN(nn.Module):
    def __init__(self, num_classes=39):  
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                    
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789') + ['@', '#', '$', '&']

    def configure_model(self, configs):
        self.optimizer = optim.Adam(self.parameters(), lr=configs.learning_rate)
        return self

    def train(self, df: DataFrame, model, path=None):
        # Convert Spark DataFrame to numpy arrays
        data = np.array(df.select("image").collect()).reshape(-1, 1, 32, 32)
        labels = np.array(df.select("label").collect()).reshape(-1)

        # Convert to PyTorch tensors
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        label_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

        # Train mode
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(data_tensor)
        loss = self.criterion(outputs, label_tensor)
        loss.backward()
        self.optimizer.step()

        # Predictions & metrics
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        labels = label_tensor.cpu().numpy()

        accuracy = accuracy_score(labels, predicted)
        precision = precision_score(labels, predicted, average="macro", zero_division=0)
        recall = recall_score(labels, predicted, average="macro", zero_division=0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return [self, predicted, accuracy, loss.item(), precision, recall, f1]

    def predict(self, df: DataFrame, model, path=None):
        data = np.array(df.select("image").collect()).reshape(-1, 1, 32, 32)
        labels = np.array(df.select("label").collect()).reshape(-1)

        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        label_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data_tensor)
            loss = self.criterion(outputs, label_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted = predicted.cpu().numpy()
        labels = label_tensor.cpu().numpy()
        accuracy = accuracy_score(labels, predicted)
        precision = precision_score(labels, predicted, average="macro", zero_division=0)
        recall = recall_score(labels, predicted, average="macro", zero_division=0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        cm = confusion_matrix(labels, predicted, labels=np.arange(len(self.classes)))

        self.visualize(data, predicted, labels)

        return [predicted, accuracy, loss.item(), precision, recall, f1, cm]

    def visualize(self, images: np.ndarray, predicted_labels: np.ndarray, true_labels: np.ndarray):
        os.makedirs('images', exist_ok=True)
        images = images.reshape(-1, 1, 32, 32)
        images_tensor = torch.tensor(images, dtype=torch.float32)

        cluster_dict = {i: [] for i in range(len(self.classes))}
        true_label_dict = {i: [] for i in range(len(self.classes))}

        for i in range(len(predicted_labels)):
            pred_class = predicted_labels[i]
            cluster_dict[pred_class].append(images_tensor[i])
            true_label_dict[pred_class].append(true_labels[i])

        for i in range(len(self.classes)):
            if len(cluster_dict[i]) > 0:
                cluster_images = torch.stack(cluster_dict[i][:10])
                true_labels_for_class = true_label_dict[i][:10]
                grid = make_grid(cluster_images, nrow=5, padding=2)
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(grid.permute(1, 2, 0).numpy().squeeze(), cmap='gray')
                plt.title(f"Predicted: {self.classes[i]} | True: {[self.classes[tl] for tl in true_labels_for_class]}")
                plt.axis('off')
                safe_class = self.classes[i].replace('@', 'at').replace('#', 'hash').replace('$', 'dollar').replace('&', 'and')
                plt.savefig(f"images/predicted_class_{safe_class}.png")
                plt.close()