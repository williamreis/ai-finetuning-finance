from transformers import pipeline

pipe = pipeline("text-classification", model="./lora-model")
print(pipe("Banco do Brasil anunciou queda de 20% no lucro trimestral."))
