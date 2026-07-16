Reimplementation of Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. “Neural Machine Translation by Jointly Learning to Align and Translate.” arXiv:1409.0473. Preprint, arXiv, May 19, 2016. https://doi.org/10.48550/arXiv.1409.0473.

Builds on seq2seq by allowing each decoder state to take a weighted sum of each of the encoder states rather than just the final encoder state, allowing the decoder to learn how to "pay attention" to each encoder state.

For educational purposes.
