from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import numpy as np

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=20)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# To apply the preprocessing fnction to tokenize the text and truncate sequence only till maximum input length
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True , max_length = 512 , padding = True)

tokenized_20NG = ds.map(preprocess_function,batched = True)

train_encodings = tokenized_20NG['train']
test_encodings = tokenized_20NG['test']

print(train_encodings)
print(test_encodings)

train_labels = torch.tensor(dataset['train']['label'])
test_labels = torch.tensor(dataset['test']['label'])

print(train_labels)
print(test_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The map function outputs as a Dataset object to need to convert into Pytorch tensor (Rember)
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])

train_dataset = TensorDataset(
    train_input_ids,
    train_attention_mask,
    train_labels
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_function = torch.nn.CrossEntropyLoss()

epochs = 4
for epoch in range(epochs):  
    model.train()
    total_loss = 0
    
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        loss = loss_function(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {average_loss:.4f}\n")


model.save_pretrained("./bert-20newsgroups")
tokenizer.save_pretrained("./bert-20newsgroups")