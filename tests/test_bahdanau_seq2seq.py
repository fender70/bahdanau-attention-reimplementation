import torch

from bahdanau.model import Bahdanau, Decoder, Encoder, Seq2Seq

SRC_VOCAB_SIZE = 10
TGT_VOCAB_SIZE = 12
BATCH_SIZE = 4
SRC_LENGTH = 3
TGT_LENGTH = 5
EMBEDDING_DIM = 32
ENCODER_HIDDEN_DIM = 48
DECODER_HIDDEN_DIM = 64
ATTENTION_DIM = 20
NUM_LAYERS = 2
SOS_IDX = 1

def build_components() -> tuple[Encoder, Bahdanau, Decoder, Seq2Seq]:
    encoder = Encoder(
        vocab_size=SRC_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        encoder_hidden_dim=ENCODER_HIDDEN_DIM,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )
    attention = Bahdanau(
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
        encoder_hidden_dim=ENCODER_HIDDEN_DIM,
        attention_dim=ATTENTION_DIM,
    )
    decoder = Decoder(
        vocab_size=TGT_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        encoder_hidden_dim=ENCODER_HIDDEN_DIM,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        bahdanau=attention,
        target_vocab_size=TGT_VOCAB_SIZE,
    )
    return encoder, attention, decoder, model


def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.randint(
        low=3,
        high=SRC_VOCAB_SIZE,
        size=(BATCH_SIZE, SRC_LENGTH),
    )
    target_body = torch.randint(
        low=3,
        high=TGT_VOCAB_SIZE,
        size=(BATCH_SIZE, TGT_LENGTH - 1),
    )
    sos = torch.full((BATCH_SIZE, 1), SOS_IDX, dtype=torch.long)
    tgt = torch.cat((sos, target_body), dim=1)
    return src, tgt


def test_attention_shapes_and_normalization() -> None:
    encoder, attention, _, _ = build_components()
    src, _ = make_batch()

    encoder_outputs, decoder_hidden, _ = encoder(src)
    context, weights = attention(decoder_hidden[-1], encoder_outputs)

    assert context.shape == (BATCH_SIZE, ENCODER_HIDDEN_DIM)
    assert weights.shape == (BATCH_SIZE, SRC_LENGTH)
    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(BATCH_SIZE, device=weights.device),
        atol=1e-6,
    )


def test_decoder_step_shapes() -> None:
    encoder, attention, decoder, _ = build_components()
    src, _ = make_batch()

    encoder_outputs, hidden, cell = encoder(src)
    context, _ = attention(hidden[-1], encoder_outputs)
    input_token = torch.full((BATCH_SIZE,), SOS_IDX, dtype=torch.long)

    logits, next_hidden, next_cell = decoder(
        input_token,
        context,
        hidden,
        cell,
    )

    assert logits.shape == (BATCH_SIZE, TGT_VOCAB_SIZE)
    assert next_hidden.shape == (NUM_LAYERS, BATCH_SIZE, DECODER_HIDDEN_DIM)
    assert next_cell.shape == (NUM_LAYERS, BATCH_SIZE, DECODER_HIDDEN_DIM)


def test_seq2seq_forward_and_generate_shapes() -> None:
    _, _, _, model = build_components()
    src, tgt = make_batch()

    logits = model(src, tgt)
    generated = model.generate(src, sos_idx=SOS_IDX, max_length=TGT_LENGTH)

    assert logits.shape == (BATCH_SIZE, TGT_LENGTH - 1, TGT_VOCAB_SIZE)
    assert generated.shape == (BATCH_SIZE, TGT_LENGTH)
    assert generated.dtype == torch.long


def test_gradients_reach_attention_parameters() -> None:
    _, attention, _, model = build_components()
    src, tgt = make_batch()

    loss = model(src, tgt).sum()
    loss.backward()

    for parameter in attention.parameters():
        assert parameter.grad is not None
