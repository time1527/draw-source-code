import torch
from transformers.models.mixtral import MixtralModel,MixtralConfig

def run():
    config = MixtralConfig(hidden_size=120,
                         intermediate_size=240,
                         num_hidden_layers=3,
                         num_attention_heads=20,
                         num_key_value_heads = 10,
                         num_experts_per_tok = 2,
                         max_position_embeddings=100,
                         attn_implementation = "eager")
    model = MixtralModel(config=config)
    inputs_ids = torch.randint(0,config.vocab_size,size = (2,7))
    res = model(inputs_ids)
    print(res)


if __name__ == "__main__":
    run()