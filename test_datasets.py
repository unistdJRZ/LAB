from datasets import load_dataset, Image

dataset = load_dataset("G:/datasets/food101", split="train")
print(dataset[0]["image"])