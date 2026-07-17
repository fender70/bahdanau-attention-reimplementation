import torch
from torch import nn


class Bahdanau(nn.Module):
    """Additive attention over encoder outputs."""

    def __init__(
        self,
        decoder_hidden_dim: int,
        encoder_hidden_dim: int,
        attention_dim: int,
    ) -> None:
        super().__init__()

        self.project_query = nn.Linear(
            in_features=decoder_hidden_dim,
            out_features=attention_dim,
        )
        self.project_encoder_outputs = nn.Linear(
            in_features=encoder_hidden_dim,
            out_features=attention_dim,
        )
        self.project_energy = nn.Linear(
            in_features=attention_dim,
            out_features=1,
        )

    def forward(
        self,
        query: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_query = self.project_query(query).unsqueeze(1)
        projected_encoder_outputs = self.project_encoder_outputs(encoder_outputs)

        energy = torch.tanh(projected_query + projected_encoder_outputs)
        scores = self.project_energy(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)

        context = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)
        return context, weights
