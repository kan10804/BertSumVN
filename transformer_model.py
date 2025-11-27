# transformer_model.py
# BertAbsSum model (BERT encoder + Transformer decoder)
# Decoder uses separate token embedding + sinusoidal positional embedding (recommended).

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig
from transformer_utils import timing, get_logger

# Use existing transformer implementation in your repo
from transformer.Layers import DecoderLayer
from transformer.Models2 import (
    get_non_pad_mask,
    get_sinusoid_encoding_table,
    get_attn_key_pad_mask,
    get_subsequent_mask
)

logger = get_logger(__name__)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, hidden_size):
        super().__init__()
        table = get_sinusoid_encoding_table(
            n_position=max_seq_len + 1,  # include padding idx 0
            d_hid=hidden_size,
            padding_idx=0
        )
        # table is torch.Tensor (n_position x d_hid)
        self.embedding = nn.Embedding.from_pretrained(table, freeze=True)

    def forward(self, pos_ids):
        # pos_ids: (B, T) with values in [0, max_seq_len]
        return self.embedding(pos_ids)


class BertDecoder(nn.Module):
    def __init__(self, config, constants, device):
        super().__init__()
        self.device = device
        self.constants = constants

        dec = config["decoder_config"]
        d_model = dec["d_model"]
        vocab_size = dec["vocab_size"]
        n_layers = dec["n_layers"]

        # token embedding (separate from BERT embeddings)
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=constants.PAD)

        # sinusoidal positional embedding
        self.pos_embedding = SinusoidalPositionEmbedding(
            max_seq_len=constants.MAX_TGT_SEQ_LEN,
            hidden_size=d_model
        )

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=dec["d_model"],
                d_inner=dec["d_inner"],
                n_head=dec["n_head"],
                d_k=dec["d_k"],
                d_v=dec["d_v"]
            ) for _ in range(n_layers)
        ])

        # final projection to vocab
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    @timing
    def forward(self, src_seq, enc_output, tgt_seq):
        """
        src_seq: (B, S) - used to create enc_mask
        enc_output: (B, S_enc, D_enc)
        tgt_seq: (B, T) token ids (includes BOS at idx 0 of sequence when used)
        returns: logits (B, T, V)
        """
        tgt_seq = tgt_seq.to(self.device)
        src_seq = src_seq.to(self.device)

        batch_size, tgt_len = tgt_seq.size()

        # position ids start at 0 .. tgt_len-1
        pos = torch.arange(0, tgt_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # masks
        # pad mask for decoder self-attention: (B, T, T) created by helper
        pad_mask = get_attn_key_pad_mask(key_seq=tgt_seq, query_seq=tgt_seq)  # expects (key_seq, query_seq)
        sub_mask = get_subsequent_mask(tgt_seq)  # subsequent mask (B, T, T)
        dec_mask = (pad_mask | sub_mask)  # boolean mask

        # encoder->decoder attention mask: keys are encoder tokens, queries are tgt tokens
        enc_mask = get_attn_key_pad_mask(key_seq=src_seq, query_seq=tgt_seq)  # (B, T, S_enc)

        # non-pad mask for post-attention feed-forward (B, T, 1)
        non_pad = get_non_pad_mask(tgt_seq)

        # embeddings
        tok_emb = self.token_embed(tgt_seq)  # (B, T, d_model)
        pos_emb = self.pos_embedding(pos)    # (B, T, d_model)
        out = tok_emb + pos_emb

        # pass through decoder layers
        for layer in self.layers:
            out, _, _ = layer(out, enc_output, non_pad, dec_mask, enc_mask)

        logits = self.linear(out)  # (B, T, V)
        return logits, None


class BertAbsSum(nn.Module):
    """
    BERT encoder + Transformer decoder for abstractive summarization.
    config: dict with "bert_model", "bert_config", "decoder_config", "freeze_encoder" keys
    constants: object with PAD/BOS/EOS/MAX_TGT_SEQ_LEN
    device: 'cpu' or 'cuda'
    """
    def __init__(self, config, constants, device):
        super().__init__()
        self.config = config
        self.device = device
        self.constants = constants

        # encoder from transformers
        self.encoder = AutoModel.from_pretrained(config["bert_model"])

        # optional freeze encoder
        if config.get("freeze_encoder", False):
            for p in self.encoder.parameters():
                p.requires_grad = False

        # decoder
        self.decoder = BertDecoder(config, constants, device)

        # if encoder hidden size != decoder d_model, add projection
        enc_hid_size = getattr(self.encoder.config, "hidden_size", None)
        dec_d_model = config["decoder_config"]["d_model"]
        if enc_hid_size is None:
            enc_hid_size = self.encoder.config.hidden_size  # fallback

        if enc_hid_size != dec_d_model:
            # project encoder outputs to decoder dimension
            self.enc_to_dec_proj = nn.Linear(enc_hid_size, dec_d_model, bias=False)
            logger.info(f"Added encoder->decoder projection: {enc_hid_size} -> {dec_d_model}")
        else:
            self.enc_to_dec_proj = None

    def batch_encode_src_seq(self, src, mask):
        # src: (B, S), mask: (B, S)
        outputs = self.encoder(input_ids=src, attention_mask=mask, return_dict=True)
        last_hidden = outputs.last_hidden_state  # (B, S, D_enc)
        if self.enc_to_dec_proj is not None:
            last_hidden = self.enc_to_dec_proj(last_hidden)  # (B, S, D_dec)
        return last_hidden

    def forward(self, batch_src_seq, batch_src_mask, batch_tgt_seq, batch_tgt_mask):
        # Training forward: returns logits (B, T, V)
        enc_out = self.batch_encode_src_seq(batch_src_seq, batch_src_mask)
        logits, _ = self.decoder(batch_src_seq, enc_out, batch_tgt_seq)
        return logits

    @timing
    def greedy_decode(self, batch):
        """
        batch: tuple like (raw, src_ids, src_mask, None, None)
        returns: dec_seq (B, T_generated)
        """
        src = batch[1].to(self.device)
        mask = batch[2].to(self.device)

        enc_out = self.batch_encode_src_seq(src, mask)

        # start with BOS token
        dec_seq = torch.full((src.size(0), 1),
                             self.constants.BOS,
                             dtype=torch.long).to(self.device)

        finished = torch.zeros(src.size(0), dtype=torch.bool, device=self.device)

        for _ in range(self.constants.MAX_TGT_SEQ_LEN):
            logits, _ = self.decoder(src, enc_out, dec_seq)  # (B, T, V)
            next_tok = logits[:, -1].argmax(-1).unsqueeze(1)  # (B,1)
            dec_seq = torch.cat([dec_seq, next_tok], dim=1)  # append

            # update finished
            finished = finished | (next_tok.squeeze(1) == self.constants.EOS)
            if finished.all():
                break

        return dec_seq
