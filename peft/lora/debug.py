from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

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

if __name__ == "__main__":
    run1()