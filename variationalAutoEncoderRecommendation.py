import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class VAE(nn.Module):
    def __init__(self, num_items, hidden_dim=200, latent_dim=50):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.mean(h), self.log_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(model, optimizer, data, epochs=100, batch_size=128):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch)
            loss = loss_function(recon_batch, batch, mu, log_var)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss/data.shape[0]:.4f}')

def generate_recommendations(model, user_vector, top_k=10):
    model.eval()
    with torch.no_grad():
        mu, log_var = model.encode(user_vector.unsqueeze(0))
        z = model.reparameterize(mu, log_var)
        recon = model.decode(z).squeeze()
    
    # Get top-k items
    _, indices = torch.topk(recon, top_k)
    return indices.tolist()

# Example usage
num_users, num_items = 1000, 500
data = torch.rand(num_users, num_items)  # Replace with your actual user-item interaction data

model = VAE(num_items)
optimizer = optim.Adam(model.parameters())

# Train the model
train(model, optimizer, data)

# Generate recommendations for a user
user_id = 0
user_vector = data[user_id]
recommendations = generate_recommendations(model, user_vector)
print(f"Top 10 recommendations for user {user_id}: {recommendations}")