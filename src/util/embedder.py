import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
import tensorflow_hub as hub
import numpy as np

class Embedder:
    """Embedder Class: a class to embedd tokens or sentences using w2v or bert embeddings.
    public functions:
        - embed : embeds a token or sentence using the initialized embedding function
        - get_embeddings_size: returns the size of the calculated embeddings
    instance variables:
        - embedding_type: the type of embedding, either word2vec or BERt embeddings
        - model_src: path to BERT model if BERT embeddings are to be used

    """
    def __init__(self, embedding_type: str, model_src: str):
        """
        creates an embedder instance
        Args:
        - :param embedding_type: the type of embedding, either word2vec or BERt embeddings
        - :param model_src: path to BERT model if BERT embeddings are to be used
    """
        self.embedding_type = embedding_type
        self.model_src = model_src
        self.embedding_model = None
        self.tokenizer = None
        self._create()

    def _create(self):
        """creates embedder instance using the provided embedding type and model path"""
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

    def embed(self, token: str, truncation=False) -> np.ndarray():
        """
        Embed a sentence or token with the embedder
        Args:
            - :param token: The token or sentence to embed
            - :param truncation: To use when embedding sentences. W2v does currently not trunctuate. BERT however does
        truncate to 512, if the flag is set to True.
        Return Value:
            :return: an ndarray with length 250 for w2v and 768 for BERT

        """
        if self.embedding_type == 'w2v':
            return self.embedding_model([token]).numpy()[0]
        else:
            input_ids = self.tokenizer.encode(token,
                                              is_split_into_words=False,
                                              truncation=truncation,
                                              return_tensors="pt", )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            input_ids = input_ids.to(device)
            with torch.no_grad():
                hidden_states = self.embedding_model(input_ids=input_ids)[1]
                return torch.mean(hidden_states[-1], dim=1).squeeze().to('cpu').numpy()

    def get_embedding_size(self) -> int:
        if self.embedding_type == 'w2v':
            return 250
        else:
            return 768
