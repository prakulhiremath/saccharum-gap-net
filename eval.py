import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
# Assuming these are accessible from your project structure
# from data import prepare_data
# from models.model import get_model

def evaluate_model(model_path, data_dir, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model & Best Weights
    # model = get_model(classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Get Test Data (using validation loader for this example)
    # _, test_loader, _ = prepare_data(data_dir, batch_size=32)

    all_preds = []
    all_labels = []

    print("Running Evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. Generate Metrics
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 4. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Saccharum-GAP-Net Confusion Matrix')
    plt.savefig('evaluation/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    # class_list = ['Healthy', 'Mosaic', 'Redrot', 'Rust', 'Yellow']
    # evaluate_model('models/checkpoints/best_saccharum_gap.pth', 'data/raw', class_list)
    pass
