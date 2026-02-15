import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter


from datasets import load_dataset

ds = load_dataset("SetFit/20_newsgroups")

print(ds)

train_labels = ds['train']['label_text']
print(train_labels)
label_counts = Counter(train_labels)
print(label_counts)


label_counts = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
plt.figure(figsize=(12, 6))
plt.bar(label_counts.keys(), label_counts.values(), color='steelblue')
plt.xlabel('Category')
plt.ylabel('Number of Samples')
plt.title('Training Set Class Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(r'SLM_20_Newsgroups\images', dpi=300)
plt.show()


print(f"Max: {max(label_counts.values())}, Min: {min(label_counts.values())}")
print(f"Imbalance Ratio: {max(label_counts.values()) / min(label_counts.values()):.2f}x")