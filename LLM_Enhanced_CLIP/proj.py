import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

file_path = ""
df = pd.read_csv(file_path, dtype=str)

node_ids = df.iloc[:, 0].values
features_list = []
max_dim = 0

for row in df.iloc[:, 1:].values:
    try:
        feature_values = [json.loads(f.replace("'", '"')) if isinstance(f, str) and f.startswith("[") else [] for f in row]
        feature_values = sum(feature_values, [])
        max_dim = max(max_dim, len(feature_values))
        features_list.append(feature_values)
    except:
        features_list.append([])

features_array = np.array([f + [0] * (max_dim - len(f)) for f in features_list], dtype=np.float32)

# 转换为 PyTorch 张量并移动到 GPU
features_tensor = torch.tensor(features_array, dtype=torch.float32, device=device)
print(f"节点数: {features_tensor.shape[0]}, 统一特征维度: {features_tensor.shape[1]}")

# ---------------- Soft Attention + Linear降维 + 线性解码 ----------------
class SoftAttentionWithDecoder(nn.Module):
    def __init__(self, in_size, hidden_size=256, reduced_size=128):
        super(SoftAttentionWithDecoder, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.reduce_dim = nn.Linear(in_size, reduced_size)  # 降维层
        self.reconstruct_dim = nn.Linear(reduced_size, in_size)  # 反向恢复维度

    def forward(self, z):
        w = self.project(z)  # (N, 1)
        beta = torch.softmax(w, dim=0)  # (N, 1)
        attention_out = beta * z  # (N, F)
        reduced_out = self.reduce_dim(attention_out)  # 降维 (N, reduced_size)
        reconstructed_out = self.reconstruct_dim(reduced_out)  # 重新映射回 (N, in_size)
        return reduced_out, reconstructed_out

# 超参数
hidden_size = 512
reduced_size = 64  # 降维后的维度
learning_rate = 5e-5
num_epochs = 2000

# 初始化模型并移动到 GPU
attention_model = SoftAttentionWithDecoder(in_size=features_tensor.shape[1], hidden_size=hidden_size, reduced_size=reduced_size).to(device)

# 训练设置
optimizer = optim.Adam(attention_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练 Soft Attention with Linear 降维 + 重建
for epoch in range(num_epochs):
    optimizer.zero_grad()
    reduced_features, reconstructed_features = attention_model(features_tensor)  # 获取降维和重建特征
    loss = criterion(reconstructed_features, features_tensor)  # 计算重建损失
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

print("Soft Attention with Linear (降维 + 重建) 训练完成！")

# ---------------- 获取降维后的特征并保存 ----------------

# 获取降维后的特征 (注意: [0] 表示获取降维部分，而不是重建部分)
final_reduced_features = attention_model(features_tensor)[0].detach().cpu().tolist()

final_features_df = pd.DataFrame(final_reduced_features)
final_features_df.to_csv("fused_node_features_dim0.csv", index=False)
