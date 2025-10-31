from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("bert-base-chinese")
tok.save_pretrained("models/bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")
model.save_pretrained("models/bert-base-chinese")

