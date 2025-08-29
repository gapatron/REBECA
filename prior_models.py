import torch
import torch.nn as nn
import torch.nn.functional as F



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
    

class CrossAttentionDiffusionPrior(nn.Module):
    """
    Enhanced diffusion prior with cross-attention for user preferences
    """
    def __init__(
        self,
        img_embed_dim=1024,
        num_users=94,
        num_tokens=8,
        n_heads=8,
        num_layers=6,
        dim_feedforward=2048,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = img_embed_dim // num_tokens
        
        # User and score embeddings
        self.user_embedding = nn.Embedding(num_users + 1, self.token_dim)  # +1 for CFG
        self.score_embedding = nn.Embedding(3, self.token_dim)  # 0, 1, 2 (CFG)
        self.time_embedding = SinusoidalEmbedding(self.token_dim)
        
        # Cross-attention layers for user-image interaction
        self.user_image_cross_attention = nn.MultiheadAttention(
            embed_dim=self.token_dim, num_heads=n_heads, batch_first=True, dropout=0.1
        )
        
        # Self-attention for image tokens
        self.image_self_attention = nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=n_heads, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        
        # Multi-layer transformer for image processing
        self.image_transformer = nn.TransformerEncoder(
            self.image_self_attention, num_layers=num_layers
        )
        
        # Final prediction head
        self.output_fc = nn.Linear(self.token_dim, self.token_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.token_dim)
        self.norm2 = nn.LayerNorm(self.token_dim)
    
    def forward(self, x, t, user_id, score):
        """
        Forward pass
        Args:
            x: (batch_size, img_embed_dim) - noisy image embeddings
            t: (batch_size) - timesteps
            user_id: (batch_size) - user IDs
            score: (batch_size) - binary scores (0, 1, 2 for CFG)
        Returns:
            predicted_noise: (batch_size, img_embed_dim)
        """
        bsz = x.shape[0]
        
        # Tokenize image embeddings
        img_tokens = x.view(bsz, self.num_tokens, self.token_dim)  # (bsz, num_tokens, token_dim)
        
        # Create user and score tokens
        user_token = self.user_embedding(user_id).unsqueeze(1)  # (bsz, 1, token_dim)
        score_token = self.score_embedding(score).unsqueeze(1)  # (bsz, 1, token_dim)
        
        # Handle timestep expansion if needed
        if t.dim() == 0:
            t = t.unsqueeze(0)  # Make it 1D if it's a scalar
        if t.shape[0] == 1 and bsz > 1:
            t = t.expand(bsz)  # Expand to batch size if needed
        
        time_embed = self.time_embedding(t)  # (bsz, token_dim)
        time_token = time_embed.unsqueeze(1)  # (bsz, 1, token_dim)
        
        # Ensure all tokens are on the same device
        device = x.device
        user_token = user_token.to(device)
        score_token = score_token.to(device)
        time_token = time_token.to(device)
        
        # Combine conditioning tokens
        condition_tokens = torch.cat([user_token, score_token, time_token], dim=1)  # (bsz, 3, token_dim)
        
        # Cross-attention: User preferences influence image generation
        # Query: image tokens, Key/Value: user preferences
        enhanced_img_tokens, _ = self.user_image_cross_attention(
            query=img_tokens,  # What we want to generate
            key=condition_tokens,  # User preferences
            value=condition_tokens
        )
        enhanced_img_tokens = self.norm1(enhanced_img_tokens + img_tokens)  # Residual connection
        
        # Self-attention among enhanced image tokens
        enhanced_img_tokens = self.image_transformer(enhanced_img_tokens)
        enhanced_img_tokens = self.norm2(enhanced_img_tokens)
        
        # Final prediction
        predicted_noise = self.output_fc(enhanced_img_tokens)
        
        return predicted_noise.reshape(bsz, -1)  # Back to (bsz, img_embed_dim)


class CrossAttentionDiffusionPriorLarge(nn.Module):
    """
    Large-scale cross-attention diffusion prior (~68M parameters)
    Scaled up to match original model capacity
    """
    def __init__(
        self,
        img_embed_dim=1024,
        num_users=94,
        num_tokens=1,  # Single token for larger dimensions
        n_heads=16,    # Increased from 8
        num_layers=8,  # Increased from 6
        dim_feedforward=4096,  # Increased from 2048
        hidden_dim=512,  # Reduced to match target size
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = img_embed_dim // num_tokens  # Should be 1024
        self.hidden_dim = hidden_dim
        
        # User and score embeddings (scaled up)
        self.user_embedding = nn.Embedding(num_users + 1, self.hidden_dim)  # +1 for CFG
        self.score_embedding = nn.Embedding(3, self.hidden_dim)  # 0, 1, 2 (CFG)
        self.time_embedding = SinusoidalEmbedding(self.hidden_dim)
        
        # Input projection to hidden dimension
        self.input_projection = nn.Linear(self.token_dim, self.hidden_dim)
        
        # Cross-attention layers for user-image interaction (scaled up)
        self.user_image_cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=n_heads, batch_first=True, dropout=0.1
        )
        
        # Self-attention for image tokens (scaled up)
        self.image_self_attention = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        
        # Multi-layer transformer for image processing (scaled up)
        self.image_transformer = nn.TransformerEncoder(
            self.image_self_attention, num_layers=num_layers
        )
        
        # Additional processing layers
        self.intermediate_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intermediate_norm = nn.LayerNorm(self.hidden_dim)
        
        # Final prediction head (project back to token dimension)
        self.output_fc = nn.Linear(self.hidden_dim, self.token_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, x, t, user_id, score):
        """
        Forward pass
        Args:
            x: (batch_size, img_embed_dim) - noisy image embeddings
            t: (batch_size) - timesteps
            user_id: (batch_size) - user IDs
            score: (batch_size) - binary scores (0, 1, 2 for CFG)
        Returns:
            predicted_noise: (batch_size, img_embed_dim)
        """
        bsz = x.shape[0]
        
        # Tokenize image embeddings
        img_tokens = x.view(bsz, self.num_tokens, self.token_dim)  # (bsz, 1, 1024)
        
        # Project to hidden dimension
        img_tokens = self.input_projection(img_tokens)  # (bsz, 1, 1024)
        
        # Create user and score tokens
        user_token = self.user_embedding(user_id).unsqueeze(1)  # (bsz, 1, 1024)
        score_token = self.score_embedding(score).unsqueeze(1)  # (bsz, 1, 1024)
        
        # Handle timestep expansion if needed
        if t.dim() == 0:
            t = t.unsqueeze(0)  # Make it 1D if it's a scalar
        if t.shape[0] == 1 and bsz > 1:
            t = t.expand(bsz)  # Expand to batch size if needed
        
        time_embed = self.time_embedding(t)  # (bsz, 1024)
        time_token = time_embed.unsqueeze(1)  # (bsz, 1, 1024)
        
        # Ensure all tokens are on the same device
        device = x.device
        user_token = user_token.to(device)
        score_token = score_token.to(device)
        time_token = time_token.to(device)
        
        # Combine conditioning tokens
        condition_tokens = torch.cat([user_token, score_token, time_token], dim=1)  # (bsz, 3, 1024)
        
        # Cross-attention: User preferences influence image generation
        # Query: image tokens, Key/Value: user preferences
        enhanced_img_tokens, _ = self.user_image_cross_attention(
            query=img_tokens,  # What we want to generate
            key=condition_tokens,  # User preferences
            value=condition_tokens
        )
        enhanced_img_tokens = self.norm1(enhanced_img_tokens + img_tokens)  # Residual connection
        
        # Self-attention among enhanced image tokens
        enhanced_img_tokens = self.image_transformer(enhanced_img_tokens)
        enhanced_img_tokens = self.norm2(enhanced_img_tokens)
        
        # Additional processing
        enhanced_img_tokens = self.intermediate_fc(enhanced_img_tokens)
        enhanced_img_tokens = self.norm3(enhanced_img_tokens)
        
        # Final prediction (project back to token dimension)
        predicted_noise = self.output_fc(enhanced_img_tokens)
        
        return predicted_noise.reshape(bsz, -1)  # Back to (bsz, img_embed_dim)



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


class CrossAttentionPerLayerDiffusionPrior(nn.Module):
    """
    Diffusion prior with cross-attention after each layer
    This demonstrates the alternative architecture with cross attention throughout
    """
    def __init__(
        self,
        img_embed_dim=1024,
        num_users=94,
        num_tokens=8,
        n_heads=8,
        num_layers=6,
        dim_feedforward=2048,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = img_embed_dim // num_tokens
        self.num_layers = num_layers
        
        # User and score embeddings
        self.user_embedding = nn.Embedding(num_users + 1, self.token_dim)  # +1 for CFG
        self.score_embedding = nn.Embedding(3, self.token_dim)  # 0, 1, 2 (CFG)
        self.time_embedding = SinusoidalEmbedding(self.token_dim)
        
        # Create layers with cross attention after each
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'self_attention': nn.TransformerEncoderLayer(
                    d_model=self.token_dim, nhead=n_heads, 
                    dim_feedforward=dim_feedforward,
                    batch_first=True, dropout=0.1
                ),
                'cross_attention': nn.MultiheadAttention(
                    embed_dim=self.token_dim, num_heads=n_heads, 
                    batch_first=True, dropout=0.1
                ),
                'norm1': nn.LayerNorm(self.token_dim),
                'norm2': nn.LayerNorm(self.token_dim),
                'norm3': nn.LayerNorm(self.token_dim),
            })
            self.layers.append(layer)
        
        # Final prediction head
        self.output_fc = nn.Linear(self.token_dim, self.token_dim)
    
    def forward(self, x, t, user_id, score):
        """
        Forward pass with cross attention after each layer
        """
        bsz = x.shape[0]
        
        # Tokenize image embeddings
        img_tokens = x.view(bsz, self.num_tokens, self.token_dim)
        
        # Create conditioning tokens
        user_token = self.user_embedding(user_id).unsqueeze(1)
        score_token = self.score_embedding(score).unsqueeze(1)
        
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and bsz > 1:
            t = t.expand(bsz)
        
        time_embed = self.time_embedding(t)
        time_token = time_embed.unsqueeze(1)
        
        # Combine conditioning tokens
        condition_tokens = torch.cat([user_token, score_token, time_token], dim=1)
        
        # Process through layers with cross attention after each
        current_tokens = img_tokens
        
        for i, layer in enumerate(self.layers):
            # Self attention
            attended = layer['self_attention'](current_tokens)
            current_tokens = layer['norm1'](current_tokens + attended)
            
            # Cross attention with conditioning
            cross_attended, _ = layer['cross_attention'](
                query=current_tokens,
                key=condition_tokens,
                value=condition_tokens
            )
            current_tokens = layer['norm2'](current_tokens + cross_attended)
            
            # Final normalization
            current_tokens = layer['norm3'](current_tokens)
        
        # Final prediction
        predicted_noise = self.output_fc(current_tokens)
        
        return predicted_noise.reshape(bsz, -1)

class TimestepEmbed(nn.Module):
    def __init__(self, dim, fourier_dim=256):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, t):
        # t: (B,) integer or float timesteps in [0, T)
        half = self.fourier_dim // 2
        freqs = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(1.0)),
                torch.log(torch.tensor(10000.0)),
                steps=half, device=t.device
            )
        )
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return self.proj(emb)  # (B, dim)

class AdaLNZero(nn.Module):
    """Adaptive LayerNorm with zero-initialized scale/shift heads."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale = nn.Linear(cond_dim, hidden_dim)
        self.to_shift = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)
    def forward(self, x, c):
        x = self.norm(x)
        scale = self.to_scale(c).unsqueeze(1)  # (B,1,H)
        shift = self.to_shift(c).unsqueeze(1)
        return x * (1 + scale) + shift

class GatedCrossAttn(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(1))  # start closed
    def forward(self, x, kv):
        # x: (B, S, H) query; kv: (B, C, H) key/value (conditioning)
        y, _ = self.attn(query=x, key=kv, value=kv, need_weights=False)
        return x + self.gate * y

class LearnedTokenizer(nn.Module):
    """Project 1D embedding <-> tokens instead of .view()"""
    def __init__(self, embed_dim, n_tokens):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.token_dim = embed_dim // n_tokens
        assert embed_dim % n_tokens == 0
        self.to_tokens = nn.Linear(embed_dim, embed_dim)   # (B,E)->(B,E) then reshape
        self.from_tokens = nn.Linear(embed_dim, embed_dim) # detokenizer after concat
    def split(self, x):
        z = self.to_tokens(x)  # (B,E)
        return z.view(x.size(0), self.n_tokens, self.token_dim)  # (B,T,H)
    def merge(self, tokens):
        z = tokens.reshape(tokens.size(0), -1)  # (B,E)
        return self.from_tokens(z)  # (B,E)

class PriorBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, cond_dim, mlp_mult=4):
        super().__init__()
        self.adaln1 = AdaLNZero(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.adaln2 = AdaLNZero(hidden_dim, cond_dim)
        self.cross_attn = GatedCrossAttn(hidden_dim, n_heads)
        self.adaln3 = AdaLNZero(hidden_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_mult*hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_mult*hidden_dim, hidden_dim),
        )
        # zero-init last linear for residual safety
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, x, cond_vec, cond_tokens):
        # cond_vec: (B, cond_dim) summarized conditioning (time+user+score)
        h = self.adaln1(x, cond_vec)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.adaln2(x, cond_vec)
        x = self.cross_attn(h, cond_tokens)        # gated cross-attn
        h = self.adaln3(x, cond_vec)
        x = x + self.mlp(h)                         # gated via zero-init
        return x

class RebecaDiffusionPrior(nn.Module):
    """
    Drop-in replacement: learned tokenizer + AdaLN-Zero + gated per-block cross-attn
    """
    def __init__(self,
                 img_embed_dim=1280,
                 num_users=210,
                 num_tokens=8,
                 hidden_dim=None,
                 n_heads=8,
                 num_layers=6,
                 score_classes=2):
        super().__init__()
        hidden_dim = hidden_dim or (img_embed_dim // num_tokens)
        assert img_embed_dim % num_tokens == 0
        self.tok = LearnedTokenizer(img_embed_dim, num_tokens)
        self.hidden_dim = hidden_dim
        self.n_tokens = num_tokens
        self.token_dim = img_embed_dim // num_tokens

        # Project token_dim -> hidden_dim (optional widening)
        self.in_proj  = nn.Linear(self.token_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.token_dim)

        # Conditioning
        self.user_emb  = nn.Embedding(num_users + 1, hidden_dim)       # last index = null user (CFG)
        self.score_emb = nn.Embedding(score_classes + 1, hidden_dim)        # include CFG/null
        self.t_embed   = TimestepEmbed(hidden_dim)

        # small MLP to summarize cond tokens -> cond_vec
        self.cond_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PriorBlock(hidden_dim, n_heads, cond_dim=hidden_dim) for _ in range(num_layers)
        ])

        # prediction head with schedule-aware scaling (see notes below)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def build_cond(self, t, user_id, score):
        u = self.user_emb(user_id)     # (B,H)
        s = self.score_emb(score)      # (B,H)
        te = self.t_embed(t)           # (B,H)
        cond_tokens = torch.stack([u, s, te], dim=1)   # (B,3,H)
        cond_vec = self.cond_pool(cond_tokens.mean(dim=1))  # (B,H)
        return cond_vec, cond_tokens

    def forward(self, x, t, user_id, score):
        B = x.size(0)
        img_tokens = self.tok.split(x)                  # (B,T,Htok)
        img_tokens = self.in_proj(img_tokens)           # (B,T,H)
        cond_vec, cond_tokens = self.build_cond(t, user_id, score)
        h = img_tokens
        for blk in self.blocks:
            h = blk(h, cond_vec, cond_tokens)
        h = self.head(h)
        h = self.out_proj(h)                            # (B,T,Htok)
        y = self.tok.merge(h)                           # (B,E)
        return y



class HierarchicalCrossAttentionPrior(nn.Module):
    """
    Hierarchical cross attention that groups embedding dimensions semantically
    Instead of arbitrary slicing, groups related dimensions together
    """
    def __init__(
        self,
        img_embed_dim=1024,
        num_users=94,
        n_heads=8,
        num_groups=8,  # Number of semantic groups
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.img_embed_dim = img_embed_dim
        self.num_groups = num_groups
        self.group_dim = img_embed_dim // num_groups
        
        # User and conditioning embeddings
        self.user_embedding = nn.Embedding(num_users + 1, img_embed_dim)
        self.score_embedding = nn.Embedding(3, img_embed_dim)
        self.time_embedding = SinusoidalEmbedding(img_embed_dim)
        
        # Time projection
        self.time_projection = nn.Sequential(
            nn.Linear(img_embed_dim, img_embed_dim * 4),
            nn.GELU(),
            nn.Linear(img_embed_dim * 4, img_embed_dim)
        )
        
        # Group projection layers (learn to group related dimensions)
        self.group_projections = nn.ModuleList([
            nn.Linear(img_embed_dim, self.group_dim) for _ in range(num_groups)
        ])
        
        # Cross attention for each group
        self.group_cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.group_dim,
                num_heads=n_heads,
                batch_first=True,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Group self attention layers
        self.group_self_attentions = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.group_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        # Global cross attention (across all groups)
        self.global_cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=img_embed_dim,
                num_heads=n_heads,
                batch_first=True,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Layer normalizations
        self.group_cross_norms = nn.ModuleList([
            nn.LayerNorm(self.group_dim) for _ in range(num_layers)
        ])
        self.group_self_norms = nn.ModuleList([
            nn.LayerNorm(self.group_dim) for _ in range(num_layers)
        ])
        self.global_cross_norms = nn.ModuleList([
            nn.LayerNorm(img_embed_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(img_embed_dim, img_embed_dim * 2),
            nn.GELU(),
            nn.Linear(img_embed_dim * 2, img_embed_dim)
        )
    
    def forward(self, x, t, user_id, score):
        """
        Forward pass with hierarchical cross attention
        Groups embedding dimensions semantically, then applies cross attention
        """
        bsz = x.shape[0]
        
        # Create conditioning embeddings
        user_emb = self.user_embedding(user_id)
        score_emb = self.score_embedding(score)
        time_emb = self.time_embedding(t)
        time_emb = self.time_projection(time_emb)
        
        # Combine conditioning
        condition_emb = user_emb + score_emb + time_emb
        
        # Project to groups (learned semantic grouping)
        x_groups = []
        condition_groups = []
        
        for i in range(self.num_groups):
            x_group = self.group_projections[i](x)  # (bsz, group_dim)
            condition_group = self.group_projections[i](condition_emb)  # (bsz, group_dim)
            
            x_groups.append(x_group.unsqueeze(1))  # (bsz, 1, group_dim)
            condition_groups.append(condition_group.unsqueeze(1))  # (bsz, 1, group_dim)
        
        # Process through layers
        for layer_idx in range(len(self.group_cross_attentions)):
            # Process each group with cross attention
            for group_idx in range(self.num_groups):
                # Cross attention within group
                cross_attended, _ = self.group_cross_attentions[layer_idx](
                    query=x_groups[group_idx],
                    key=condition_groups[group_idx],
                    value=condition_groups[group_idx]
                )
                x_groups[group_idx] = self.group_cross_norms[layer_idx](
                    x_groups[group_idx] + cross_attended
                )
                
                # Self attention within group
                self_attended = self.group_self_attentions[layer_idx](x_groups[group_idx])
                x_groups[group_idx] = self.group_self_norms[layer_idx](
                    x_groups[group_idx] + self_attended
                )
            
            # Global cross attention across all groups
            # Concatenate all groups
            x_global = torch.cat(x_groups, dim=1)  # (bsz, num_groups, group_dim)
            x_global = x_global.view(bsz, -1)  # (bsz, img_embed_dim)
            x_global = x_global.unsqueeze(1)  # (bsz, 1, img_embed_dim)
            
            condition_global = condition_emb.unsqueeze(1)  # (bsz, 1, img_embed_dim)
            
            global_cross_attended, _ = self.global_cross_attentions[layer_idx](
                query=x_global,
                key=condition_global,
                value=condition_global
            )
            x_global = self.global_cross_norms[layer_idx](x_global + global_cross_attended)
            
            # Redistribute back to groups
            x_global = x_global.squeeze(1)  # (bsz, img_embed_dim)
            for group_idx in range(self.num_groups):
                x_groups[group_idx] = self.group_projections[group_idx](x_global).unsqueeze(1)
        
        # Final output: combine all groups
        output = torch.cat(x_groups, dim=1)  # (bsz, num_groups, group_dim)
        output = output.view(bsz, -1)  # (bsz, img_embed_dim)
        predicted_noise = self.output_projection(output)
        
        return predicted_noise


class AdaptiveCrossAttentionPrior(nn.Module):
    """
    Adaptive cross attention that learns the best conditioning strategy
    Uses attention to determine how to condition the embedding
    """
    def __init__(
        self,
        img_embed_dim=1024,
        num_users=94,
        n_heads=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.img_embed_dim = img_embed_dim
        
        # User and conditioning embeddings
        self.user_embedding = nn.Embedding(num_users + 1, img_embed_dim)
        self.score_embedding = nn.Embedding(3, img_embed_dim)
        self.time_embedding = SinusoidalEmbedding(img_embed_dim)
        
        # Time projection
        self.time_projection = nn.Sequential(
            nn.Linear(img_embed_dim, img_embed_dim * 4),
            nn.GELU(),
            nn.Linear(img_embed_dim * 4, img_embed_dim)
        )
        
        # Adaptive conditioning layers
        self.adaptive_layers = nn.ModuleList([
            nn.ModuleDict({
                'conditioning_attention': nn.MultiheadAttention(
                    embed_dim=img_embed_dim,
                    num_heads=n_heads,
                    batch_first=True,
                    dropout=dropout
                ),
                'feature_attention': nn.MultiheadAttention(
                    embed_dim=img_embed_dim,
                    num_heads=n_heads,
                    batch_first=True,
                    dropout=dropout
                ),
                'feature_mlp': nn.Sequential(
                    nn.Linear(img_embed_dim, dim_feedforward),
                    nn.GELU(),
                    nn.Linear(dim_feedforward, img_embed_dim)
                ),
                'norm1': nn.LayerNorm(img_embed_dim),
                'norm2': nn.LayerNorm(img_embed_dim),
                'norm3': nn.LayerNorm(img_embed_dim)
            }) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(img_embed_dim, img_embed_dim * 2),
            nn.GELU(),
            nn.Linear(img_embed_dim * 2, img_embed_dim)
        )
    
    def forward(self, x, t, user_id, score):
        """
        Forward pass with adaptive cross attention
        Learns how to best condition the embedding
        """
        bsz = x.shape[0]
        
        # Create conditioning embeddings
        user_emb = self.user_embedding(user_id)
        score_emb = self.score_embedding(score)
        time_emb = self.time_embedding(t)
        time_emb = self.time_projection(time_emb)
        
        # Combine conditioning
        condition_emb = user_emb + score_emb + time_emb
        
        # Reshape for attention
        x_tokens = x.unsqueeze(1)  # (bsz, 1, img_embed_dim)
        condition_tokens = condition_emb.unsqueeze(1)  # (bsz, 1, img_embed_dim)
        
        # Process through adaptive layers
        current_x = x_tokens
        
        for layer in self.adaptive_layers:
            # Adaptive conditioning: learn how to condition
            conditioned, _ = layer['conditioning_attention'](
                query=current_x,
                key=condition_tokens,
                value=condition_tokens
            )
            current_x = layer['norm1'](current_x + conditioned)
            
            # Feature attention: refine the conditioned features
            refined, _ = layer['feature_attention'](
                query=current_x,
                key=current_x,
                value=current_x
            )
            current_x = layer['norm2'](current_x + refined)
            
            # Feature MLP: further processing
            mlp_out = layer['feature_mlp'](current_x)
            current_x = layer['norm3'](current_x + mlp_out)
        
        # Final output
        output = current_x.squeeze(1)
        predicted_noise = self.output_projection(output)
        
        return predicted_noise
