import torch
import torch.nn as nn



############ Auto Encoder is CLIP


class AE(nn.Module):
    def __init__(self, image_size=64, in_channels=3, latent_dim=10):
        super().__init__()
        assert image_size == 64, "This architecture assumes 64x64 images"

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),   # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),            # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),           # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU()           # 4x4
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)               # (B, 256, 1, 1)
        self.latent_head = nn.Conv2d(256, latent_dim, 1)         # (B, latent_dim, 1, 1)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),    # 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),     # 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),      # 32x32
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1), nn.Sigmoid()  # 64x64
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)           # (B, 256, 1, 1)
        z = self.latent_head(x)           # (B, latent_dim, 1, 1)
        return z.squeeze(-1).squeeze(-1)  # (B, latent_dim)

    def decode(self, z):
        x = self.fc_dec(z)                # (B, 256*4*4)
        x = x.view(-1, 256, 4, 4)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.exp(torch.arange(half_size, dtype=torch.float32, device=x.device) * 
                        -(torch.log(torch.tensor(10000.0, device=x.device)) / (half_size - 1)))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size
    

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out
    

class SimulationREBECAModel(nn.Module):
    def __init__(self, img_embed_dim=16, num_users=4, user_embed_dim=16, time_dim=100, 
                 score_dim=16, hidden_dim=512, num_res_blocks=2):
        super().__init__()
        self.time_embedding = SinusoidalEmbedding(time_dim)
        self.score_embedding = nn.Embedding(2, score_dim)
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.input_fc = nn.Linear(img_embed_dim + user_embed_dim + time_dim + score_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.output_fc = nn.Linear(hidden_dim, img_embed_dim)
        
    def forward(self, x, t, user_id, score):
        t_embed = self.time_embedding(t)
        score_embed = self.score_embedding(score)
        user_embedding = self.user_embedding(user_id)
        net_input = torch.cat([x, user_embedding, t_embed, score_embed], dim=-1)
        out = self.input_fc(net_input)
        out = self.input_bn(out)
        out = self.activation(out)
        for block in self.res_blocks:
            out = block(out)
        output = self.output_fc(out)
        if x.shape == output.shape:
            output = x + output
        return output