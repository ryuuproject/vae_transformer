# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class SMILESDataset(Dataset):
    def __init__(self, data, char_to_index, max_length=120):
        self.data = data
        self.char_to_index = char_to_index
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def smiles_to_tensor(self, smiles):
        tensor = [self.char_to_index[c] for c in smiles]
        return torch.tensor(tensor, dtype=torch.long)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        smiles_tensor = self.smiles_to_tensor(smiles)
        return smiles_tensor

def load_data(file_path, max_length=120):
    with open(file_path, 'r') as f:
        smiles = [line.strip() for line in f.readlines()]
    # Truncate or pad SMILES strings to the same length
    smiles = [s.ljust(max_length, ' ') for s in smiles]
    return smiles


batch_size = 64

train_file_path = '/content/train.csv'
test_file_path = '/content/test.csv'

train_data = load_data(train_file_path)
test_data = load_data(test_file_path)
# Create the char_to_index mapping
unique_chars = set("".join(train_data) + "".join(test_data))
char_to_index = {char: i for i, char in enumerate(sorted(unique_chars))}
train_dataset = SMILESDataset(train_data, char_to_index)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = SMILESDataset(test_data, char_to_index)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=None, key_padding_mask=None)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=None, key_padding_mask=None)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class VAE(nn.Module):
    def __init__(self, vocab_size, hidden_dim, latent_dim, max_length, nhead=4, dim_feedforward=512):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Encoder
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.transformer_decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward)
        self.decoder_out = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length
        self.pos_encoder = nn.Embedding(max_length, hidden_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = self.embedding(x)
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden.squeeze(0)
        mu, log_var = self.encoder_mu(hidden), self.encoder_log_var(hidden)

        # Reparameterization
        z = self.reparameterize(mu, log_var)

        # Decoder
        positions = torch.arange(0, self.max_length, dtype=torch.long).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        pos_encoded = self.pos_encoder(positions)

        z = self.z_to_hidden(z)
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        z = z + pos_encoded
        output = self.transformer_decoder_layer(z, z)
        output = self.decoder_out(output)
        output = F.softmax(output, dim=-1)

        return output, mu, log_var

def loss_function(recon_x, x, mu, log_var):
    # Reconstruction loss (cross-entropy)
    recon_loss = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), reduction='mean')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss /= x.size(0) * x.size(1)

    return recon_loss + kl_loss

def train(model, dataloader, optimizer, device, num_epochs=50):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch)

            loss = loss_function(recon_batch, batch, mu, log_var)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(char_to_index)
hidden_dim = 256
latent_dim = 64
max_length = 120
learning_rate = 0.01
batch_size = 64
nhead = 4
dim_feedforward = 512

model = VAE(vocab_size, hidden_dim, latent_dim, max_length, nhead, dim_feedforward).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model and get the losses
losses = train(model, train_dataloader, optimizer, device)

# Plot the training loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

def generate_molecule(model, latent_vector, index_to_char, max_length=120):
    model.eval()
    with torch.no_grad():
        decoded = model.decoder(latent_vector)
        decoded_smiles = torch.argmax(decoded, dim=2)
        decoded_smiles = [index_to_char[int(x)] for x in decoded_smiles[0]]
        return ''.join(decoded_smiles).strip()

latent_vector = torch.randn(1, latent_dim).to(device)
generated_smiles = generate_molecule(model, latent_vector, index_to_char)
print("Generated SMILES: ", generated_smiles)