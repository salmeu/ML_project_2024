import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensuring that the GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class CustomDataset(Dataset):
    def __init__(self, image_names, labels_df, transform=None):
        self.image_names = image_names
        self.labels = labels_df
        self.transform = transform
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        image_name = self.image_names.iloc[idx]
        label = self.labels.iloc[idx]
        label = torch.tensor(label.values, dtype=torch.float32)
        image = Image.open('/filepath to images/' + image_name)
        if self.transform:
            image = self.transform(image)
        return image, label
# Extract dataset and prepare directories
csv_path = "/filepath/file.csv"
extracted_images_path = "/filepath to images/"
# Load the CSV file with labels
name_and_label_df = pd.read_csv(csv_path, delimiter=',')
X = name_and_label_df['Name']
y = name_and_label_df.drop(columns=['Name'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define hyperparameters
num_epochs = 10
learning_rate = 0.001
num_classes = 3
# Load pretrained ResNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, num_classes),
    nn.Sigmoid()  # For multi-label binary classification
)
model = model.to(device)
transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = CustomDataset(X_train, y_train, transform=transform)
test_dataset = CustomDataset(X_test, y_test, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training loop with GPU support
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.4f}")
            running_loss = 0.0
print("Training complete.")
# Save the fine-tuned model
torch.save(model.state_dict(), "resnet_blinker_model.pth")

def calculate_accuracy(real_labels, predicted_labels):
  L_accuracy = accuracy_score(real_labels[:, 0], predicted_labels[:, 0])
  B_accuracy = accuracy_score(real_labels[:, 1], predicted_labels[:, 1])
  R_accuracy = accuracy_score(real_labels[:, 2], predicted_labels[:, 2])
  return L_accuracy, B_accuracy, R_accuracy

def evaluate_model(model, dataloader, threshold=0.5, device="cpu"):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move to device
            # Forward pass
            outputs = model(images)
            # Apply threshold to get binary predictions
            predictions = (outputs > threshold).float()
            all_predictions.append(predictions.cpu().numpy())  # Move predictions to CPU for metrics
            all_labels.append(labels.cpu().numpy())  # Move labels to CPU for metrics
    # Combine all batches
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    # Calculate evaluation metrics
    accuracy = calculate_accuracy(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average=None)
    recall = recall_score(all_labels, all_predictions, average=None)
    f1 = f1_score(all_labels, all_predictions, average=None)
    print(f"Evaluation Metrics (Threshold: {threshold}):")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    return all_predictions, all_labels

# Evaluate the model and visualize results
predictions, labels = evaluate_model(model, test_dataloader, threshold=0.5, device=device)

#Visualizing multi-class confusion matrix
def confusion_matrix(labels, predictions):
    ml_cm = multilabel_confusion_matrix(labels, predictions)

    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5))

    # Class names for clarity (optional, update as per your classes)
    class_names = ['Left Blinker', 'Brake', 'Right Blinker']

    for i, (cm, ax) in enumerate(zip(ml_cm, axes)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Orange', cbar=False, ax=ax)
        ax.set_title(f'{class_names[i]} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Positive', 'Negative'])
        ax.set_yticklabels(['Positive', 'Negative'])

    plt.tight_layout()
    plt.show()
