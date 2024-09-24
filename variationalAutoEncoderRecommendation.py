import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class VAE(nn.Module):
    def __init__(self, num_items, hidden_dim=200, latent_dim=50):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential( # sequential is a container that holds a sequence of layers
            nn.Linear(num_items, hidden_dim), # input layer
            nn.ReLU(), # activation function
            nn.Linear(hidden_dim, hidden_dim), # hidden layer
            nn.ReLU() # activation function
        )
        
        self.mean = nn.Linear(hidden_dim, latent_dim) # mean layer
        self.log_var = nn.Linear(hidden_dim, latent_dim) # log variance layer
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), # input layer                
            nn.ReLU(), # activation function
            nn.Linear(hidden_dim, hidden_dim), # hidden layer
            nn.ReLU(), # activation function
            nn.Linear(hidden_dim, num_items), # output layer
            nn.Sigmoid() # activation function
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.mean(h), self.log_var(h) # return the mean and log variance
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # standard deviation
        eps = torch.randn_like(std) # random noise
        return mu + eps * std # return the latent vector
    
    def decode(self, z):
        return self.decoder(z) # decode the latent vector to get the reconstructed user vector
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var # return the reconstructed user vector, the mean and log variance

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') # binary cross entropy loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # Kullback-Leibler divergence
    return BCE + KLD # return the total loss

def train(model, optimizer, data, epochs=100, batch_size=128):
    model.train() # set the model to training mode
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
    model.eval() # set the model to evaluation mode
    with torch.no_grad(): # no grad because we are not training the model
        mu, log_var = model.encode(user_vector.unsqueeze(0)) # encode the user vector
        z = model.reparameterize(mu, log_var) # reparameterize the latent vector
        recon = model.decode(z).squeeze() # decode the latent vector to get the reconstructed user vector
    
    # Get top-k items
    _, indices = torch.topk(recon, top_k) # get the top-k items
    return indices.tolist() # return the top-k items

# Example usage
num_users, num_items = 1000, 500
data = torch.rand(num_users, num_items)  # Replace with your actual user-item interaction data

model = VAE(num_items) # create the model   
optimizer = optim.Adam(model.parameters()) # create the optimizer

# Train the model
train(model, optimizer, data)

# Generate recommendations for a user
user_id = 0
user_vector = data[user_id]
recommendations = generate_recommendations(model, user_vector)
print(f"Top 10 recommendations for user {user_id}: {recommendations}")