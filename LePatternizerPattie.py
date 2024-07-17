import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Define the Transformer model for vision
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=1, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim

        self.to_patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        p = self.patch_size
        x = self.to_patch_embedding(img)
        x = x.flatten(2).transpose(1, 2)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

# Define the dataset class
class PuzzleDataset(Dataset):
    def __init__(self, examples, target_shape):
        self.examples = examples
        self.target_shape = target_shape
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((target_shape[0], target_shape[1])),
            transforms.ConvertImageDtype(torch.float32)
        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_grid, output_grid = self.examples[idx]
        input_grid = normalize_and_reshape(np.array(input_grid), self.target_shape)
        output_grid = normalize_and_reshape(np.array(output_grid), self.target_shape)
        return (self.transform(input_grid),
                self.transform(output_grid).view(-1))

# Normalize and reshape the data to a fixed size
def normalize_and_reshape(data, target_shape):
    data = np.array(data)
    norm_data = np.zeros(target_shape)
    min_shape = min(data.shape[0], target_shape[0]), min(data.shape[1], target_shape[1])
    norm_data[:min_shape[0], :min_shape[1]] = data[:min_shape[0], :min_shape[1]]
    return norm_data

# Define the reward function
def reward_func(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)
    match = np.sum((y_true == y_pred_rounded).astype(int))
    total_elements = y_true.size
    reward = (match / total_elements) * 100  # Scale reward to percentage of match
    return reward

# Training function with learning rate adjustment
def train_model(model, train_loader, epochs, initial_learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, verbose=True)  # More dynamic learning rate scheduler

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_reward = 0

        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            reward = reward_func(targets.numpy(), outputs.detach().numpy())
            total_reward += reward

        avg_loss = total_loss / len(train_loader)
        avg_reward = total_reward / len(train_loader)
        scheduler.step(avg_loss)  # Adjust learning rate based on loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}, Avg Reward: {avg_reward}")

# Test the trained network with a specific input
def test_model(model, test_input, target_shape):
    model.eval()
    test_input = normalize_and_reshape(test_input, target_shape).flatten()
    test_input = torch.tensor(test_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    output = model(test_input).detach().numpy().reshape(target_shape)
    print("Test Output:")
    print(np.round(output))

# Expanded examples based on the provided images
examples = [
    (
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 3, 0, 0, 0, 0],
            [0, 3, 3, 3, 1, 3, 3, 3, 0],
            [0, 3, 1, 1, 1, 1, 1, 3, 0],
            [0, 3, 1, 1, 1, 1, 1, 3, 0],
            [3, 1, 1, 1, 2, 1, 1, 1, 3],
            [0, 3, 1, 1, 1, 1, 1, 3, 0],
            [0, 3, 1, 1, 1, 1, 1, 3, 0],
            [0, 3, 3, 3, 1, 3, 3, 3, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 0]
        ]
    ),
    (
        [
            [0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0]
        ],
        [
            [0, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1, 0, 1]
        ]
    ),
    (
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ],
        [
            [1, 1, 2, 2],
            [4, 4, 5, 5],
            [8, 8, 9, 9],
            [12, 12, 13, 13]
        ]
    ),
    (
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ],
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ]
    )
]

# Create dataset and dataloader
target_shape = (9, 9)  # Adjust based on the largest grid size
dataset = PuzzleDataset(examples, target_shape)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)  # Increase batch size for more efficient training

# Initialize and train the model
input_channels = 1
output_channels = target_shape[0] * target_shape[1]
model = VisionTransformer(img_size=9, patch_size=3, num_classes=output_channels, dim=512, depth=120, heads=16, mlp_dim=4096)
train_model(model, train_loader, epochs=1000, initial_learning_rate=0.001)

# Test the trained model with a specific input
test_input = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]
test_model(model, test_input, target_shape)
