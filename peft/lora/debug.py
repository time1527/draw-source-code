# reference code: https://github.com/huggingface/peft/blob/main/README.md
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model,AutoPeftModelForCausalLM

def run1():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", low_cpu_mem_usage=True)

    config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                        inference_mode=False, 
                        r=8, 
                        lora_alpha=32, 
                        lora_dropout=0.1,
                        layers_to_transform = 0)

    model = get_peft_model(model, config)
    model.print_trainable_parameters()


def run2():
    import torch

    model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    model.eval()
    inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

    # 断点1
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
    # 断点2:transformers/models/opt/modeling_opt.py/OptAtttention/forward/...self.q_proj...
    # 断点3:transformers/models/opt/modeling_opt.py/OptAtttention/forward/...self.v_proj...
    # 断点4:peft/tuners/lora.py/Linear/forward
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

if __name__ == "__main__":
    # run1() # lora1
    run2() # lora2