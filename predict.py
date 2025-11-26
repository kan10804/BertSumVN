import torch
from transformers import AutoTokenizer
from transformer_model import BertAbsSum
from params_helper import Params

# BẮT BUỘC: Config Params để không lỗi
Params.mode = "eval"
Params.max_src_len = 256
Params.max_tgt_len = 40
Params.block_ngram_repeat = 2
Params.min_tgt_len = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========== LOAD TOKENIZER ============
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# =========== LOAD MODEL ============
def load_model(ckpt="output/checkpoint-1/pytorch_model.bin"):
    print("Loading model from:", ckpt)

    config = {
        "bert_model": "vinai/phobert-base",
        "freeze_encoder": True,
        "bert_config": {"hidden_size": 768},
        "decoder_config": {
            "n_layers": 4,
            "n_head": 8,
            "d_k": 32,
            "d_v": 32,
            "d_model": 768,
            "d_inner": 1024,
            "vocab_size": len(tokenizer)
        }
    }

    model = BertAbsSum(config, device).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    print("Model loaded!")
    return model

model = load_model()

# =========== TÓM TẮT ============
def summarize(text):
    src_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    src_mask = torch.ones_like(src_ids).to(device)

    batch = (
        ["id01"],
        src_ids,
        src_mask,
        torch.zeros((1, 10)).long().to(device),
        torch.zeros((1, 10)).long().to(device)
    )

    result = model.beam_decode(
        batch_guids=batch[0],
        batch_src_seq=batch[1],
        batch_src_mask=batch[2],
        beam_size=3,
        n_best=1
    )

    token_ids = result[0][0][1]
    summary = tokenizer.decode(token_ids, skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    text = "Bộ Y tế vừa công bố báo cáo mới về tình hình dịch bệnh tại Việt Nam..."
    print("SUMMARY:", summarize(text))
