import torch
from torch import nn

from .bahdanau import Bahdanau
from .decoder import Decoder
from .encoder import Encoder


class Seq2Seq(nn.Module):
    """Coordinates encoding, attention, and autoregressive decoding."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        bahdanau: Bahdanau,
        target_vocab_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bahdanau = bahdanau
        self.target_vocab_size = target_vocab_size

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        batch_size = src.shape[0]
        target_length = tgt.shape[1]

        encoder_outputs, hidden, cell = self.encoder(src)
        input_token = tgt[:, 0]

        all_logits = torch.empty(
            batch_size,
            target_length - 1,
            self.target_vocab_size,
            device=src.device,
            dtype=self.decoder.output.weight.dtype,
        )

        for t in range(1, target_length):
            context, _ = self.bahdanau(hidden[-1], encoder_outputs)
            logits, hidden, cell = self.decoder(
                input_token,
                context,
                hidden,
                cell,
            )
            all_logits[:, t - 1, :] = logits
            input_token = tgt[:, t]

        return all_logits

    def generate(
        self,
        src: torch.Tensor,
        sos_idx: int,
        max_length: int,
    ) -> torch.Tensor:
        batch_size = src.shape[0]
        input_token = torch.full(
            (batch_size,),
            sos_idx,
            dtype=torch.long,
            device=src.device,
        )

        encoder_outputs, hidden, cell = self.encoder(src)
        generated_steps: list[torch.Tensor] = []

        for _ in range(max_length):
            context, _ = self.bahdanau(hidden[-1], encoder_outputs)
            logits, hidden, cell = self.decoder(
                input_token,
                context,
                hidden,
                cell,
            )
            input_token = logits.argmax(dim=-1)
            generated_steps.append(input_token)

        return torch.stack(generated_steps, dim=1)
