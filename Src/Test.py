from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDatase
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


model = BertForSequenceClassification.from_pretrained("./bert-20newsgroups")
tokenizer = BertTokenizer.from_pretrained("./bert-20newsgroups")
model.to(device)
model.eval()

test_encodings = tokenizer(
    list(dataset['test']['text']),
    truncation = True,
    padding = True,
    max_length = 512,
    return_tensors = 'pt'
    )

test_labels = torch.tensor(dataset['test']['label'])
test_dataset = TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask'],
    test_labels
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(len(test_labels))


all_predictions = []
all_true_labels = []

with torch.no_grad():
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(test_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        # Get predicted labels
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(labels.numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Processed {(batch_idx + 1) * 32} / {len(test_labels)} samples")

all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

accuracy = accuracy_score(all_true_labels, all_predictions)
precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)

print(f"Accuracy (weighted):  {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted):    {recall:.4f}")
print(f"F1-Score (weighted):  {f1:.4f}")


# Confusion matrix

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - 20 Newsgroups Test Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# predict new_text 
def predict_text(text: str):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
    
    # Get class & confidence
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()
    
    # Get label name from dataset
    sample = dataset['train'][0]  # Get one sample to see structure
    # Use the predicted class as index in label_text
    all_labels = set(dataset['train']['label_text'])
    label_name = list(all_labels)[predicted_class]
    
    return label_name, confidence