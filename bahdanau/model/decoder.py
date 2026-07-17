import torch
from torch import nn


class Decoder(nn.Module):
    """One recurrent decoder step conditioned on an attention context."""

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
            input_size=embedding_dim + encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(
            in_features=decoder_hidden_dim,
            out_features=vocab_size,
        )

    def forward(
        self,
        input_token: torch.Tensor,
        context: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_token).unsqueeze(1)
        context = context.unsqueeze(1)
        lstm_input = torch.cat((embedded, context), dim=-1)

        decoder_output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell),
        )
        logits = self.output(decoder_output.squeeze(1))

        return logits, hidden, cell
