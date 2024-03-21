# reference code: https://github.com/zyds/transformers-code/tree/master/01-Getting%20Started/02-pipeline
from transformers import pipeline
def run(prompt):
    pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese",device=0)
    pipe(prompt)

if __name__ == "__main__":
    run(["我觉得不太行！","我很喜欢"]) # pipeline 1
    run("我觉得不太行") # pipeline 2