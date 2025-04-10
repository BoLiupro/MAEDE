import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Soft Attention module ----------------
class SoftAttention(nn.Module):
    def __init__(self, in_size, hidden_size=256):
        super(SoftAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)  # (N, 1)
        beta = torch.softmax(w, dim=0)  # (N, 1)
        return beta * z  # (N, F)

# ---------------- AutoEncoder ----------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, reduced_dim=128):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(reduced_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

file_path = ""
df = pd.read_csv(file_path, dtype=str)  # 读取为字符串类型

node_ids = df.iloc[:, 0].values
features_list = []
max_dim = 0

# Analyze feature data
for row in df.iloc[:, 1:].values:
    try:
        feature_values = [json.loads(f.replace("'", '"')) if isinstance(f, str) and f.startswith("[") else [] for f in row]
        feature_values = sum(feature_values, [])
        max_dim = max(max_dim, len(feature_values))
        features_list.append(feature_values)
    except:
        features_list.append([])

features_array = np.array([f + [0] * (max_dim - len(f)) for f in features_list], dtype=np.float32)

features_tensor = torch.tensor(features_array, dtype=torch.float32, device=device)


hidden_size = 512
learning_rate = 1e-10
num_epochs = 1000

attention_model = SoftAttention(in_size=features_tensor.shape[1], hidden_size=hidden_size).to(device)

optimizer = optim.Adam(attention_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# train
for epoch in range(num_epochs):
    optimizer.zero_grad()
    fused_features = attention_model(features_tensor)
    loss = criterion(fused_features, features_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Soft Attention - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.10f}")

print("Soft Attention train finish！")

# Obtain processed features
soft_attention_features = attention_model(features_tensor).detach()


reduced_size = 256

# initialize AutoEncoder
autoencoder = AutoEncoder(input_dim=soft_attention_features.shape[1], reduced_dim=reduced_size).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# train AutoEncoder
num_epochs = 20
for epoch in range(num_epochs):
    optimizer.zero_grad()
    encoded, decoded = autoencoder(soft_attention_features)
    loss = criterion(decoded, soft_attention_features)  # 让重建结果尽可能接近输入
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"AutoEncoder - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

print("AutoEncoder train finish！")

# ----------------Retrieve the reduced dimensional features and save them ----------------
reduced_features = autoencoder.encoder(soft_attention_features).detach().cpu().tolist()
# print(reduced_features[0].shape)
pd.DataFrame(reduced_features, index=node_ids).to_csv("")
