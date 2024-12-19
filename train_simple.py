import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import sentencepiece as spm
from transformer_model import FridayTransformer
from dataset import FridayDataset
import pandas as pd
from sklearn.utils import shuffle

def populate_lists_from_csv(file_path, num_samples=None, output_file='sample.csv'):
    """
    Reads a CSV file, shuffles the data, and populates source and target sentence lists.

    Parameters:
        file_path (str): Path to the CSV file. The file should have 'task', 'input', and 'output' columns.
        num_samples (int): Number of samples to extract. If None, use the entire dataset.

    Returns:
        source_sentences (list): List of [task, input] pairs.
        target_sentences (list): List of corresponding outputs.
        :param output_file:
    """
    # Load the dataset
    dataset = pd.read_csv(file_path)

    # Validate the required columns
    required_columns = {'task', 'input', 'output'}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Shuffle the dataset
    dataset = shuffle(dataset).reset_index(drop=True)

    # Select a subset of rows if num_samples is specified
    if num_samples:
        dataset = dataset.head(num_samples)

    if output_file:
        dataset.to_csv(output_file, index=False)

    # Populate the lists
    source_sentences = dataset[['task', 'input']].values.tolist()
    target_sentences = dataset['output'].tolist()

    return source_sentences, target_sentences

tokenizer_path = 'models/tokenizer/model.model'
# Maximum length for padding
max_len = 350

sp_model = spm.SentencePieceProcessor()
sp_model.Load(tokenizer_path)

dataset_path = 'dataset/compiled_dataset.csv'

source_sentences, target_sentences = populate_lists_from_csv(dataset_path, 100)

# Create dataset
dataset = FridayDataset(source_sentences, target_sentences, sp_model, max_len)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
embed_dim = 512
vocab_size = sp_model.vocab_size()
num_layers = 2
n_heads = 8
learning_rate = 0.0001
num_epochs = 40
model = FridayTransformer(embed_dim=embed_dim,
                    src_vocab_size=vocab_size,
                    target_vocab_size=vocab_size,
                    num_layers=num_layers,
                    n_heads=n_heads)
criterion = nn.CrossEntropyLoss(ignore_index=sp_model.pad_id())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
def train(model, dataloader, criterion, optimizer, num_epochs, save_path):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (src, trg) in enumerate(dataloader):
            src = src
            trg = trg
            # Remove last token from target to get trg_input
            trg_input = trg[:, :-1]
            # Get predictions
            preds = model(src, trg_input)
            # Reshape predictions and targets for loss calculation
            preds = preds.reshape(-1, preds.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            # Calculate loss
            loss = criterion(preds, trg)
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
train(model, dataloader, criterion, optimizer, num_epochs, 'friday-torch.pth')