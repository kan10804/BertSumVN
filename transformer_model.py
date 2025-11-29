# transformer_model.py
import torch.nn as nn
import torch
import operator
from torch.nn.functional import log_softmax
from transformer.Layers import DecoderLayer
from transformer.Models2 import get_non_pad_mask, get_sinusoid_encoding_table, get_attn_key_pad_mask, get_subsequent_mask
from transformers import AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformer_utils import *
from params_helper import Params, Constants

logger = get_logger(__name__)

class BertPositionEmbedding(nn.Module):
	def __init__(self, max_seq_len, hidden_size, padding_idx=0):
		super().__init__()
		sinusoid_encoding = get_sinusoid_encoding_table(
			n_position=max_seq_len + 1,
			d_hid=hidden_size,
			padding_idx=padding_idx)
		self.embedding = nn.Embedding.from_pretrained(embeddings=sinusoid_encoding, freeze=True)

	def forward(self, x):
		return self.embedding(x)


class BertDecoder(nn.Module):
	def __init__(self, config, device, dropout=0.1):
		super().__init__()

		self.device = device
		bert_config = BertConfig.from_dict(config['bert_config'])
		decoder_config = config['decoder_config']
		n_layers = decoder_config['n_layers']
		n_head = decoder_config['n_head']
		d_k = decoder_config['d_k']
		d_v = decoder_config['d_v']
		d_model = decoder_config['d_model']
		d_inner = decoder_config['d_inner']
		vocab_size = decoder_config['vocab_size']

		# NOTE: Use attribute names that match checkpoint keys:
		# decoder.seq_embedding.* , decoder.pos_embedding.* , decoder.layers.* , decoder.last_linear.*
		# Many checkpoints expect a "seq_embedding" module that mimics BertEmbeddings structure.
		# To preserve compatibility while keeping control, we instantiate BertEmbeddings under seq_embedding.
		# This preserves keys like seq_embedding.word_embeddings.weight etc.
		self.seq_embedding = BertEmbeddings(config=bert_config)

		# position embedding stored under decoder.pos_embedding.embedding.weight in checkpoint
		self.pos_embedding = BertPositionEmbedding(max_seq_len=Constants.MAX_TGT_SEQ_LEN, hidden_size=d_model)

		# layers name matches checkpoint
		self.layers = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

		# final linear naming: use last_linear to match some variants; keep last_linear name in checkpoint list earlier
		self.last_linear = nn.Linear(in_features=d_model, out_features=vocab_size)

	@timing
	def forward(self, batch_src_seq, batch_enc_output, batch_tgt_seq):
		# ensure device
		batch_tgt_seq = batch_tgt_seq.to(self.device)
		if batch_enc_output is not None and batch_enc_output.device != self.device:
			batch_enc_output = batch_enc_output.to(self.device)

		dec_slf_attn_list, dec_enc_attn_list = [], []

		# -- Prepare masks
		dec_non_pad_mask = get_non_pad_mask(batch_tgt_seq)

		slf_attn_mask_subseq = get_subsequent_mask(batch_tgt_seq)
		slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=batch_tgt_seq, seq_q=batch_tgt_seq)
		dec_slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

		dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=batch_src_seq, seq_q=batch_tgt_seq)

		batch_size, tgt_seq_len = batch_tgt_seq.size()
		batch_tgt_pos = torch.arange(1, tgt_seq_len + 1).unsqueeze(0).repeat(batch_size, 1).to(self.device)

		# Use seq_embedding (BertEmbeddings) to get token embeddings (keeps compatibility with checkpoint)
		# and add our sinusoidal position embedding stored in pos_embedding
		token_emb = self.seq_embedding(batch_tgt_seq)   # [batch, seq, hidden]
		pos_emb = self.pos_embedding(batch_tgt_pos)    # [batch, seq, hidden]
		dec_output = token_emb + pos_emb

		# -- Forward through decoder layers
		for dec_layer in self.layers:
			dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
				dec_output, batch_enc_output,
				non_pad_mask=dec_non_pad_mask,
				slf_attn_mask=dec_slf_attn_mask,
				dec_enc_attn_mask=dec_enc_attn_mask)

			dec_slf_attn_list.append(dec_slf_attn)
			dec_enc_attn_list.append(dec_enc_attn)

		batch_logits = self.last_linear(dec_output)

		return batch_logits, dec_enc_attn_list


class BertAbsSum(nn.Module):
	def __init__(self, config, device):
		super().__init__()

		self.device = device
		self.config = config
		self.encoder = AutoModel.from_pretrained(config['bert_model'])

		# Freeze encoder if requested
		if config.get('freeze_encoder', False) == True:
			for param in self.encoder.parameters():
				param.requires_grad = False

		self.decoder = BertDecoder(config=config, device=device)

		# Count total params
		stats = self.get_model_stats()
		enc_params = stats['enc_params']
		dec_params = stats['dec_params']
		total_params = stats['total_params']
		logger.info(f'Encoder total parameters: {enc_params:,}')
		logger.info(f'Decoder total parameters: {dec_params:,}')
		logger.info(f'Total model parameters: {total_params:,}')
		
		if getattr(Params, 'mode', None) == 'train':
			enc_trainable_params = stats['enc_trainable_params']
			dec_trainable_params = stats['dec_trainable_params']
			total_trainable_params = stats['total_trainable_params']
			logger.info(f'Encoder trainable parameters: {enc_trainable_params:,}')
			logger.info(f'Decoder trainable parameters: {dec_trainable_params:,}')
			logger.info(f'Total trainable parameters: {total_trainable_params:,}')

	def get_model_stats(self):
		enc_params = sum(p.numel() for p in self.encoder.parameters())
		dec_params = sum(p.numel() for p in self.decoder.parameters())
		total_params = enc_params + dec_params

		enc_trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
		dec_trainable_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
		total_trainable_params = enc_trainable_params + dec_trainable_params

		stats = {
			'enc_params': enc_params,
			'dec_params': dec_params,
			'total_params': total_params,
			'enc_trainable_params': enc_trainable_params,
			'dec_trainable_params': dec_trainable_params,
			'total_trainable_params': total_trainable_params,
		}

		return stats

	# @timing
	def forward(self, batch_src_seq, batch_src_mask, batch_tgt_seq, batch_tgt_mask):
		# src/tgt shape: (batch_size, seq_len)

		# shift right (decoder input)
		batch_tgt_seq = batch_tgt_seq[:, :-1]
		batch_tgt_mask = batch_tgt_mask[:, :-1]

		# encode source
		batch_enc_output = self.batch_encode_src_seq(batch_src_seq=batch_src_seq, batch_src_mask=batch_src_mask)  							# [batch_size, seq_len, hidden_size]	
		batch_logits, _ = self.decoder.forward(batch_src_seq=batch_src_seq, batch_enc_output=batch_enc_output, batch_tgt_seq=batch_tgt_seq)	# [batch_size, seq_len, vocab_size]
		return batch_logits

	def batch_encode_src_seq(self, batch_src_seq, batch_src_mask):
		# Use window to scan the full input sequence (max 256) if the model is PhoBERT and the input data is longer than 256 tokens
		if 'phobert' in getattr(Params, 'bert_model', '').lower() and getattr(Params, 'max_src_len', 0) > 256:
			window_size = 256
			batch_src_seq1 = batch_src_seq[:,:window_size]
			batch_src_seq2 = batch_src_seq[:,window_size:]
			batch_src_mask1 = batch_src_mask[:,:window_size]
			batch_src_mask2 = batch_src_mask[:,window_size:]
			
			batch_enc_output1 = self.encoder.forward(input_ids=batch_src_seq1, attention_mask=batch_src_mask1)[0]	# [batch_size, window_size, hidden_size]
			batch_enc_output2 = self.encoder.forward(input_ids=batch_src_seq2, attention_mask=batch_src_mask2)[0]	# [batch_size, window_size, hidden_size]
			batch_enc_output = torch.cat([batch_enc_output1, batch_enc_output2], dim=1)								# [batch_size, full_seq_len, hidden_size]
		else:
			batch_enc_output = self.encoder.forward(input_ids=batch_src_seq, attention_mask=batch_src_mask)[0]  	# [batch_size, seq_len, hidden_size]

		return batch_enc_output

	@timing
	def greedy_decode(self, batch):
		# Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
		batch_src_seq = batch[1].to(self.device)
		batch_src_mask = batch[2].to(self.device)

		batch_enc_output = self.batch_encode_src_seq(batch_src_seq=batch_src_seq, batch_src_mask=batch_src_mask)	# [batch_size, seq_len, hidden_size]	
		batch_dec_seq = torch.full((batch_src_seq.size(0),), Constants.BOS, dtype=torch.long)
		batch_dec_seq = batch_dec_seq.unsqueeze(-1).type_as(batch_src_seq).to(self.device)

		for i in range(Constants.MAX_TGT_SEQ_LEN):
			output_logits, _ = self.decoder.forward(batch_src_seq=batch_src_seq, batch_enc_output=batch_enc_output, batch_tgt_seq=batch_dec_seq)
			# FIX: take logits of last timestep then argmax
			logits_last = output_logits[:, -1, :]    # shape: (batch, vocab)
			dec_output = logits_last.argmax(dim=-1) # shape: (batch,)
			batch_dec_seq = torch.cat((batch_dec_seq, dec_output.unsqueeze(-1)), 1)

			# If all beams predicted EOS, break early
			if (dec_output == Constants.EOS).all():
				break

		return batch_dec_seq

	@timing
	def beam_decode(self, batch_guids, batch_src_seq, batch_src_mask, beam_size, n_best):
		# unchanged logic, but ensure device placement
		batch_size = len(batch_guids)
		batch_src_seq = batch_src_seq.to(self.device)
		batch_src_mask = batch_src_mask.to(self.device)

		batch_enc_output = self.batch_encode_src_seq(batch_src_seq=batch_src_seq, batch_src_mask=batch_src_mask)	# [batch_size, seq_len, hidden_size]
		decoded_batch = []

		# Decoding goes through each sample in the batch
		for idx in range(batch_size):
			logger.debug(f'Decoding sample {batch_guids[idx]}')
			beam_src_seq = batch_src_seq[idx].unsqueeze(0).to(self.device)  # Batch with 1 sample
			beam_enc_output = batch_enc_output[idx].unsqueeze(0).to(self.device)   # Batch with 1 sample

			beams = []
			start_node = BeamSearchNode(prev_node=None, token_id=Constants.BOS, log_prob=0)
			beams.append((start_node.eval(), start_node))

			end_nodes = []

			# Start decoding process for each source sequence
			for step in range(Constants.MAX_TGT_SEQ_LEN):
				logger.debug(f'Decoding step {step} with {len(beams)} beams')
				candidates = []

				for score, node in beams:
					dec_seq = node.seq_tokens  # [id_1, id_2]
					beam_dec_seq = torch.LongTensor(dec_seq).unsqueeze(0).to(self.device)  # Batch with 1 sample

					# Decode for one step using decoder
					logger.debug('Getting decoder logits')
					output_logits, output_attentions = self.decoder.forward(batch_src_seq=beam_src_seq, batch_enc_output=beam_enc_output, batch_tgt_seq=beam_dec_seq)
					log_probs = log_softmax(output_logits[:, -1, :].squeeze(0), dim=-1)	# shape: (vocab)
					sorted_log_probs, sorted_indices = torch.sort(log_probs, dim=-1, descending=True)
					logger.debug('Logits sorted by log probs')

					# Collect top beam_size candidates for this beam instance
					candidate_count = 0
					i = 0

					while candidate_count < beam_size:
						decoded_token = sorted_indices[i].item()
						log_prob = sorted_log_probs[i].item()
						i += 1

						next_node = BeamSearchNode(prev_node=node, token_id=decoded_token, log_prob=node.log_prob + log_prob)

						# Block ngram repeats (kept original logic)
						if Params.block_ngram_repeat > 0:
							ngrams = set()
							has_repeats = False
							gram = []
							for j in range(len(next_node.seq_tokens)):
								gram = (gram + [next_node.seq_tokens[j]])[-Params.block_ngram_repeat:]
								if tuple(gram) in ngrams:
									has_repeats = True
									break
								else:
									ngrams.add(tuple(gram))
							if has_repeats:
								penaltized_log_prob = next_node.log_prob + (-10e20)
								next_node.set_log_prob(penaltized_log_prob)

						if decoded_token == Constants.EOS:
							if Params.min_tgt_len > 0:
								if next_node.seq_len >= Params.min_tgt_len:
									end_nodes.append((next_node.eval(), next_node))
							else:
								end_nodes.append((next_node.eval(), next_node))
						else:
							candidates.append((next_node.eval(), next_node))
							candidate_count += 1

				if len(end_nodes) >= n_best:
					break

				sorted_candidates = sorted(candidates, key=operator.itemgetter(0), reverse=True)
				beams = []
				for i in range(min(beam_size, len(sorted_candidates))):
					beams.append(sorted_candidates[i])

			# finalize hypotheses
			best_hypotheses = []
			sorted_beams = sorted(beams, key=operator.itemgetter(0), reverse=True)

			if len(end_nodes) < n_best:
				for i in range(min(n_best - len(end_nodes), len(sorted_beams))):
					end_nodes.append(sorted_beams[i])

			sorted_end_nodes = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)

			for i in range(min(n_best, len(sorted_end_nodes))):
				score, end_node = sorted_end_nodes[i]
				best_hypotheses.append((score, end_node.seq_tokens))

			decoded_batch.append(best_hypotheses)

		return decoded_batch


class BeamSearchNode(object):
	def __init__(self, prev_node, token_id, log_prob):
		self.finished = False
		self.prev_node = prev_node
		self.token_id = token_id
		self.log_prob = log_prob

		if prev_node is None:
			self.seq_tokens = [token_id]
		else:
			self.seq_tokens = prev_node.seq_tokens + [token_id]

		self.seq_len = len(self.seq_tokens)

		if token_id == Constants.EOS:
			self.finished = True

	def set_log_prob(self, log_prob):
		self.log_prob = log_prob

	def eval(self):
		score = self.log_prob
		norm_const = 5
		length_norm = (norm_const + self.seq_len) / (norm_const + 1.0)
		length_norm = length_norm ** getattr(Params, 'len_norm_factor', 0.0)
		score = score / length_norm
		return score

	def __lt__(self, other):
		return self.seq_len < other.seq_len

	def __gt__(self, other):
		return self.seq_len > other.seq_len
