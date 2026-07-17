import torch
from torch import nn


class Encoder(nn.Module):
    """LSTM encoder with learned bridges into the decoder state space."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=encoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.hidden_bridge = nn.Linear(
            in_features=encoder_hidden_dim,
            out_features=decoder_hidden_dim,
        )
        self.cell_bridge = nn.Linear(
            in_features=encoder_hidden_dim,
            out_features=decoder_hidden_dim,
        )

    def forward(
        self,
        src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(src)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.lstm(embedded)

        decoder_hidden = self.hidden_bridge(encoder_hidden)
        decoder_cell = self.cell_bridge(encoder_cell)

        return encoder_outputs, decoder_hidden, decoder_cell
