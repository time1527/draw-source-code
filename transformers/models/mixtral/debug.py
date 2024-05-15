import torch
from transformers.models.mixtral import MixtralModel,MixtralConfig

def run():
    config = MixtralConfig(hidden_size=120,
                         intermediate_size=240,
                         num_hidden_layers=3,
                         num_attention_heads=20,
                         num_key_value_heads = 10,
                         num_experts_per_tok = 2,
                         max_position_embeddings=100)
    model = MixtralModel(config=config)
    inputs_ids = torch.randint(0,config.vocab_size,size = (2,7))
    res = model(inputs_ids)
    print(res)


if __name__ == "__main__":
    run()

# q_slice = q[-1][-1]
# cos_slice = cos[-1][-1]
# sin_slice = sin[-1][-1]
# len = q_slice.shape[0]
# dim = q_slice.shape[1]
# res_slice = torch.zeros((len,dim))
# for i in range(len):
#     for j in range(dim):
#         if j < dim / 2:
#             res_slice[i][j] = cos_slice[i][j]*q_slice[i][j] - sin_slice[i][j]*q_slice[i][int(j+dim / 2)]
#         else:
#             # res_slice[i][j] = cos_slice[i][j]*q_slice[i][j] + sin_slice[i][j]*q_slice[i][int(j-dim / 2)]
#             res_slice[i][j] = cos_slice[i][int(j-dim / 2)]*q_slice[i][j] + sin_slice[i][int(j-dim / 2)]*q_slice[i][int(j-dim / 2)]