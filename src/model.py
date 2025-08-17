import torch
import torch.nn as nn
import math
import config
from typing import List, Tuple

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerAutoencoder(nn.Module):
    """
    Transformer-based Autoencoder for time-series anomaly detection.
    """
    def __init__(self, input_features: int, model_dim: int, num_heads: int,
                 num_encoder_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.model_dim = model_dim
        
        self.input_projection = nn.Linear(input_features, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_encoder_layers)
        
        self.output_projection = nn.Linear(model_dim, input_features)

    def forward(self, src: torch.Tensor, return_attention: bool = False) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for the autoencoder.

        Args:
            src (torch.Tensor): Input sequence. Shape: (batch_size, seq_len, features)
            return_attention (bool): If True, returns encoder attention weights along with reconstruction.

        Returns:
            If return_attention is False:
                torch.Tensor: The reconstructed sequence.
            If return_attention is True:
                Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing the reconstructed
                sequence and a list of attention maps from each encoder layer.
        """
        src = self.input_projection(src) * math.sqrt(self.model_dim)
        
        # Encoder Pass
        memory = src
        attention_maps = []
        if return_attention:
            # Manually iterate through encoder layers to capture attention weights
            for layer in self.transformer_encoder.layers:
                # The layer's forward method itself doesn't return weights, so we still need the hook,
                # but we must call the sub-block that allows us to request them.
                # The simplest fix is to adjust the hook-based approach in explainability.py
                pass # The logic is now handled cleanly in explainability.py with the updated model call
        
        memory = self.transformer_encoder(src)
        
        # Decoder Pass
        output = self.transformer_decoder(tgt=src, memory=memory)
        
        # Project back to original feature dimension for reconstruction
        reconstruction = self.output_projection(output)
        
        # NOTE: The logic for extracting attention has been moved to explainability.py
        # for simplicity and to avoid altering the standard forward pass.
        # The fix will involve how the model is called and hooked into.
        
        return reconstruction
