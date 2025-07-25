# src/model.py

import torch
import torch.nn as nn
import math
import config

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerAutoencoder(nn.Module):
    """
    Transformer-based Autoencoder for time-series anomaly detection.
    This version is modified to return encoder attention weights for explainability.
    """
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.model_dim = model_dim

        # --- Layers ---
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Final output layer
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, src, return_attention=False):
        """
        Forward pass for the Transformer Autoencoder.
        Captures attention weights using forward hooks for explainability.
        """
        attention_weights = []
        hooks = []

        if return_attention:
            def get_attention_hook(module, input, output):
                # The attention weights are the second element of the output tuple from MultiheadAttention
                attention_weights.append(output[1])
            
            # Register a forward hook on each multi-head attention layer in the encoder
            for layer in self.transformer_encoder.layers:
                hooks.append(layer.self_attn.register_forward_hook(get_attention_hook))
        
        # 1. Embed input and add positional encoding
        embedded_src = self.embedding(src) * math.sqrt(self.model_dim)
        pos_encoded_src = self.pos_encoder(embedded_src.permute(1, 0, 2)).permute(1, 0, 2)

        # 2. Pass through the encoder
        memory = self.transformer_encoder(pos_encoded_src)

        # 3. Pass through the decoder to reconstruct
        output = self.transformer_decoder(memory, memory)

        # 4. Project back to original feature dimension
        reconstructed_src = self.fc_out(output)

        # 5. Remove hooks after the forward pass is complete
        if return_attention:
            for hook in hooks:
                hook.remove()
            return reconstructed_src, attention_weights
        
        return reconstructed_src
