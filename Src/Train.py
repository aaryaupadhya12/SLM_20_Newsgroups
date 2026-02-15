import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_function = CrossEntropyLoss()

total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
print(f"Total training steps: {total_steps}")
print(f"Warmup steps: {int(0.1 * total_steps)}\n")

epochs = 3


for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_predictions = []
    all_true_labels = []
    
    # Use tqdm for progress bar
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch+1}/{epochs}",
        leave=True
    )
    
    for batch_idx, (input_ids, attention_mask, labels) in progress_bar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        # Calculate loss
        loss = loss_function(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track loss and predictions
        total_loss += loss.item()
        
        # Get predictions for accuracy calculation
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())
        
        # Update progress bar with loss
        progress_bar.set_postfix({'loss': loss.item()})
        
        if (batch_idx + 1) % 100 == 0:
            current_acc = accuracy_score(all_true_labels, all_predictions)
            print(f"\n  Batch {batch_idx+1}/{len(train_loader)}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Accuracy so far: {current_acc:.4f}")
    
    # Calculate epoch metrics
    average_loss = total_loss / len(train_loader)
    epoch_accuracy = accuracy_score(all_true_labels, all_predictions)
    
    print(f"EPOCH {epoch+1}/{epochs} COMPLETED")
    print(f"Average Loss:     {average_loss:.4f}")
    print(f"Epoch Accuracy:   {epoch_accuracy:.4f} ({epoch_accuracy*100:.2f}%)")

# Save model and tokenizer
print("Saving model and tokenizer...")
model.save_pretrained("./bert-20newsgroups")
tokenizer.save_pretrained("./bert-20newsgroups")