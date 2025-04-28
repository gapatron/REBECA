import torch
import torch.nn as nn



class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        # Scale input if necessary
        x = x * self.scale
        
        half_size = self.size // 2
        
        # Create embeddings based on the tensor device (same as x)
        emb = torch.exp(torch.arange(half_size, dtype=torch.float32, device=x.device) * 
                        -(torch.log(torch.tensor(10000.0, device=x.device)) / (half_size - 1)))
        
        # Compute the sine and cosine embeddings
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

        return emb

    def __len__(self):
        return self.size
    

class TransformerEmbeddingDiffusionModel(nn.Module):
    def __init__(
        self,
        img_embed_dim=1280,
        num_users=210,
        num_tokens=2,
        n_heads=8,
        num_layers=6,
        dim_feedforward=1024,    # FFN dimension inside transformer
    ):
        super().__init__()
        self.num_tokens = num_tokens  # num of tokens to split. e.g. if split 1280 into 2 tokens, each would have dim 640
        self.token_dim = img_embed_dim // self.num_tokens
        self.user_embedding = nn.Embedding(num_users, self.token_dim)
        self.score_embedding = nn.Embedding(2, self.token_dim)
        self.time_embedding = SinusoidalEmbedding(self.token_dim)
        # positional embedding for image tokens,
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, self.token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,  # (B, seq_len, d_model)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final linear to map to 1280
        self.output_fc = nn.Linear(self.token_dim, self.token_dim)

    def forward(self, x, t, user_id, score):
        """
        x:  [batch_size, 1280] noisy image embedding
        t:  [batch_size] timesteps
        user_id: [batch_size] user IDs
        score:   [batch_size] (0 or 1)

        Returns predicted noise with shape [batch_size, 1280].
        """
        bsz = x.shape[0]

        # 1) Split image embedding into num_tokens tokens
        #    shape => (batch_size, num_tokens, token_dim)
        img_tokens = x.view(bsz, self.num_tokens, self.token_dim)

        # 2) Create user token, score token, time token each shape => (batch_size, 1, token_dim)
        user_token = self.user_embedding(user_id).unsqueeze(1)
        score_token = self.score_embedding(score).unsqueeze(1)

        t_embed = self.time_embedding(t)  # shape: (bsz, time_embed_dim=160)
        t_token = t_embed.unsqueeze(1)    # shape: (bsz, 1, 160)

        # 3) Combine user_token, score_token, time_token with the image tokens
        #    Sequence layout: [user, score, time, image_tokens...]
        combined_tokens = torch.cat([user_token, score_token, t_token, img_tokens], dim=1)
        # combined_tokens shape: (bsz, 3+num of img token, 160)

        # 4) Add a learnable positional embedding only to the image tokens
        #    We skip positional embedding for the first 3 “conditioning” tokens.
        pos_emb = self.positional_embedding.repeat(bsz, 1, 1)  # shape: (bsz, 8, 160)
        # Insert the pos_emb
        combined_tokens[:, 3:, :] = combined_tokens[:, 3:, :] + pos_emb

        # 5) Pass through the TransformerEncoder
        #    output shape = (bsz, 3+num of img token, token_dim)
        encoded = self.transformer(combined_tokens)

        # 6) Extract the image tokens from the output
        predicted_noise_tokens = encoded[:, 3:, :]  # shape: (bsz, num of img token, token_dim)

        # pass them through a small FC for refinement
        predicted_noise_tokens = self.output_fc(predicted_noise_tokens)  # shape: (bsz, num of img token, token_dim)

        # 7) Reshape back to [batch_size, 1280]
        predicted_noise = predicted_noise_tokens.reshape(bsz, -1)  #(bsz, num of img token, token_dim) = (bsz, 1280)

        return predicted_noise
    

class TransformerEmbeddingDiffusionModelv2(nn.Module):
    def __init__(
        self,
        img_embed_dim=1280,
        num_users=210,
        n_heads=8,
        num_tokens=8,
        num_layers=6,
        dim_feedforward=4096,       # FFN dimension inside transformer
        whether_use_user_embeddings=True,
        num_user_tokens=1,          # number of tokens per user
    ):
        super().__init__()
        self.num_tokens = num_tokens        # Number of image tokens (e.g. splitting 1280 into 8 x 160)
        self.token_dim = img_embed_dim // self.num_tokens
        self.use_user_embedding = whether_use_user_embeddings
        self.num_user_tokens = num_user_tokens

        # If multiple user tokens, we embed each user into (token_dim * num_user_tokens). 
        #self.user_embedding = nn.Embedding(num_users, self.token_dim * num_user_tokens)
        #self.score_embedding = nn.Embedding(2, self.token_dim)
        # @gapatronh: I am summing one to do CFG.
        self.user_embedding = nn.Embedding(num_users + 1 , self.token_dim * num_user_tokens)
        self.score_embedding = nn.Embedding(2+1, self.token_dim)
        self.time_embedding = SinusoidalEmbedding(self.token_dim)

        # Positional embedding for the image tokens (shape: (1, num_tokens, token_dim))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.token_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,  # (B, seq_len, d_model)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final linear to map from token_dim back to token_dim
        self.output_fc = nn.Linear(self.token_dim, self.token_dim)

    def forward(self, x, t, user_id, score):
        """
        x:        [batch_size, img_embed_dim]  - noisy image embedding
        t:        [batch_size]        - timesteps
        user_id:  [batch_size]        - user IDs
        score:    [batch_size] (0 or 1)
        """
        bsz = x.shape[0]

        # 1) Split the image embedding into num_tokens tokens of shape => (batch_size, num_tokens, token_dim)
        img_tokens = x.view(bsz, self.num_tokens, self.token_dim)

        # 2) Create the score token, time token (always used)
        score_token = self.score_embedding(score).unsqueeze(1)  # (bsz, 1, token_dim)
        t_embed = self.time_embedding(t)                        # (bsz, token_dim)
        t_token = t_embed.unsqueeze(1)                          # (bsz, 1, token_dim)

        if self.use_user_embedding:
            # user_embedding(...) => (bsz, token_dim * num_user_tokens)
            user_emb = self.user_embedding(user_id)
            # Reshape to (bsz, num_user_tokens, token_dim)
            user_token = user_emb.view(bsz, self.num_user_tokens, self.token_dim)
            # We concatenate [user_token, score_token, time_token, img_tokens]
            # user_token is multiple tokens (num_user_tokens).
            tokens_to_concat = [user_token, score_token, t_token, img_tokens]
            # The first (num_user_tokens + 2) tokens are "conditioning" tokens (user + score + time).
            pos_offset = self.num_user_tokens + 2
        else:
            # If skipping user embeddings, only have [score_token, time_token] as conditioning
            tokens_to_concat = [score_token, t_token, img_tokens]
            # The first two conditioning tokens (score + time)
            pos_offset = 2

        # 4) Combine all tokens: shape => (bsz, pos_offset + num_tokens, token_dim)
        combined_tokens = torch.cat(tokens_to_concat, dim=1)

        # 5) Add learnable positional embedding only to the *image* tokens.
        #    We skip the first pos_offset "conditioning" tokens.
        #    pos_emb shape => (1, num_tokens, token_dim) repeated for batch => (bsz, num_tokens, token_dim)
        pos_emb = self.positional_embedding.repeat(bsz, 1, 1)
        combined_tokens[:, pos_offset:, :] += pos_emb

        # 6) Pass through the TransformerEncoder
        encoded = self.transformer(combined_tokens)

        # 7) Extract the image tokens from the output
        predicted_noise_tokens = encoded[:, pos_offset:, :]  # shape => (bsz, num_tokens, token_dim)

        # 8) Pass them through a small FC for refinement
        predicted_noise_tokens = self.output_fc(predicted_noise_tokens)  # (bsz, num_tokens, token_dim)

        # 9) Reshape back to [batch_size, 1280]
        predicted_noise = predicted_noise_tokens.reshape(bsz, -1)

        return predicted_noise
