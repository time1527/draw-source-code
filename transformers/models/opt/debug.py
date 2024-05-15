from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding
import torch

def run():
    num_embeddings = 5
    embedding_dim = 8
    embed = OPTLearnedPositionalEmbedding(num_embeddings,embedding_dim)
    attention_mask = torch.tensor([1,1,1,1,0,0]).view(1,-1).long()
    past_key_values_length = 2
    res = embed(attention_mask,past_key_values_length)
    """
    attention_mask = attention_mask.long()
        tensor([[1, 1, 1, 1, 0, 0]])
    positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        tensor([[ 0,  1,  2,  3, -1, -1]])
    positions = positions[:, past_key_values_length:]
        tensor([[ 2,  3, -1, -1]])
    res.size():
        torch.Size([1, 4, 8])
    """
    return res

if __name__ == "__main__":
    run()