import os
import sys
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Reconfigure stdout to use UTF-8 to prevent console encoding issues
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

print("=== Starting T5-Small GPU Fine-Tuning Setup ===")

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

# Load dataset
dataset_path = "bbc_news.csv"
if not os.path.exists(dataset_path):
    # Fallback to absolute path if running from parent
    dataset_path = "D:/PythonLearn/Data Science/News Article Classification/bbc_news.csv"

if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at bbc_news.csv or absolute path!")
    sys.exit(1)

print(f"Loading dataset from {dataset_path}...")
df = pd.read_csv(dataset_path)
df = df.dropna(subset=['description', 'title'])
print(f"Dataset loaded. Total rows: {len(df)}")

# Load model and tokenizer
t5_model_name = "t5-small"
print(f"Loading pretrained {t5_model_name} model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
model = model.to(device)

# Dataset Class
class HeadlineDataset(Dataset):
    def __init__(self, descriptions, titles, tokenizer, max_input_len=128, max_target_len=32):
        self.descriptions = descriptions.reset_index(drop=True)
        self.titles = titles.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        
    def __len__(self):
        return len(self.descriptions)
        
    def __getitem__(self, idx):
        desc = "summarize: " + str(self.descriptions[idx]).lower().strip()
        title = str(self.titles[idx]).strip()
        
        inputs = self.tokenizer(
            desc,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            title,
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        labels = targets['input_ids'].squeeze(0)
        
        # Replace padding token ids with -100 to ignore them in loss calculation
        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# DataLoader setup
print("Preparing DataLoader...")
train_dataset = HeadlineDataset(df['description'], df['title'], tokenizer)
# Batch size 16 works well on standard GPUs (e.g. 6GB+ VRAM)
# Set pin_memory=True for faster CPU-GPU transfers
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=(device.type == 'cuda'))

# Training hyperparams
epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

print(f"\nFine-tuning T5-Small on full dataset ({len(df)} samples) for {epochs} epochs...")
start_time = time.time()

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    # Use tqdm progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} Complete. Average Loss: {avg_loss:.4f}")

total_duration = time.time() - start_time
print(f"\nFine-tuning completed in {total_duration/60:.2f} minutes!")

# Save fine-tuned model and tokenizer
save_dir = "models/t5_small"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving fine-tuned model and tokenizer to '{save_dir}'...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("All model assets saved successfully!")
