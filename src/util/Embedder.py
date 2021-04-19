import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
import tensorflow_hub as hub


class Embedder:
    def __init__(self, embedding_type, model_src:str):
        self.embedding_type = embedding_type
        self.model_src = model_src
        self.embedding_model = None
        self.tokenizer = None
        self._create()

    def _create(self):
        if self.embedding_type == 'w2v':
            self.embedding_model = hub.load("https://tfhub.dev/google/Wiki-words-250-with-normalization/2")
        else:
            config = AutoConfig.from_pretrained(self.model_src,
                                                output_hidden_states=True)
            self.embedding_model = AutoModelForTokenClassification.from_pretrained(
                "/content/drive/MyDrive/Master_Arbeit/colab/SROIE_Baseline", config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/content/drive/MyDrive/Master_Arbeit/colab/SROIE_Baseline")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embedding_model.to(device)
            self.embedding_model.eval()

    def embed(self, token):
        if self.embedding_type == 'w2v':
            return self.embedding_model([token]).numpy()[0]
        else:
            input_ids = self.tokenizer.encode(token,
                                              is_split_into_words=False,
                                              truncation=False,
                                              return_tensors="pt", )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            input_ids = input_ids.to(device)
            with torch.no_grad():
                hidden_states = self.embedding_model(input_ids=input_ids)[1]
                return torch.mean(hidden_states[-1], dim=1).squeeze().to('cpu').numpy()
