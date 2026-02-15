import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

# Load dataset and model
dataset = load_dataset("SetFit/20_newsgroups")
tokenizer = BertTokenizer.from_pretrained("./bert-20newsgroups")
model = BertForSequenceClassification.from_pretrained("./bert-20newsgroups")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Get unique label names from the dataset
unique_labels = sorted(set(dataset['train']['label_text']))
label_names = unique_labels
print(f"Label names: {label_names}\n")

def predict_text(text: str):
    # Tokenize
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
    
    
    label_name = label_names[predicted_class]
    
    return label_name, confidence