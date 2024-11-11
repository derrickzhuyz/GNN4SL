from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import openai

class NLEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', vector_dim=384, openai_api_key=None):
        # Load the local SentenceTransformer model
        self.sentence_model = SentenceTransformer(model_name)
        self.vector_dim = vector_dim
        
        # Load the BERT model and tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Set OpenAI API key if provided
        if openai_api_key:
            openai.api_key = openai_api_key

    def embed_with_sentence_transformer(self, question: str) -> np.ndarray:
        # Generate the embedding using SentenceTransformer
        embedding = self.sentence_model.encode(question)
        return self._ensure_fixed_length(embedding)

    def embed_with_bert(self, question: str) -> np.ndarray:
        # Tokenize the input question
        inputs = self.bert_tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Get the embeddings from BERT
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # Get the mean of the last hidden state as the sentence embedding
        last_hidden_state = outputs.last_hidden_state
        sentence_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
        return self._ensure_fixed_length(sentence_embedding)

    def embed_with_openai(self, question: str) -> np.ndarray:
        # Call the OpenAI API to get the embedding
        response = openai.Embedding.create(input=question, model="text-embedding-ada-002")
        embedding = np.array(response['data'][0]['embedding'])
        return self._ensure_fixed_length(embedding)

    def _ensure_fixed_length(self, embedding: np.ndarray) -> np.ndarray:
        # Ensure the embedding is of fixed length
        if len(embedding) != self.vector_dim:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match the fixed dimension {self.vector_dim}.")
        return embedding

# Example usage
if __name__ == "__main__":
    # Replace 'your_openai_api_key' with your actual OpenAI API key
    embedder = NLEmbedder(openai_api_key='your_openai_api_key')
    question = "What is the capital of France?"

    # Get embeddings from different methods
    vector_sentence_transformer = embedder.embed_with_sentence_transformer(question)
    # vector_bert = embedder.embed_with_bert(question)
    # vector_openai = embedder.embed_with_openai(question)

    print("Sentence Transformer Vector:", vector_sentence_transformer)
    # print("BERT Vector:", vector_bert)
    # print("OpenAI API Vector:", vector_openai)