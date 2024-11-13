import os
import torch
import numpy as np
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


class NLEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', vector_dim=1536, openai_api_key=None, base_url=None):
        # Load the local SentenceTransformer model
        self.sentence_model = SentenceTransformer(model_name)
        self.vector_dim = vector_dim
        
        # Load the BERT model and tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Set OpenAI API key and base URL if provided
        if openai_api_key and base_url:
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=base_url,
            )

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

    def embed_with_openai_small(self, question: str) -> np.ndarray:
        # Call the OpenAI API to get the embedding
        response = self.client.embeddings.create(input=question, model="text-embedding-3-small")
        embedding = np.array(response.data[0].embedding)
        print('cost:', response.usage.total_tokens * 0.020 / 1000000) # $0.020 / 1M tokens
        return self._ensure_fixed_length(embedding)
    
    def embed_with_openai_large(self, question: str) -> np.ndarray:
        # Call the OpenAI API to get the embedding
        response = self.client.embeddings.create(input=question, model="text-embedding-3-large")
        embedding = np.array(response.data[0].embedding)
        print('cost:', response.usage.total_tokens * 0.130 / 1000000) # $0.130 / 1M tokens
        return self._ensure_fixed_length(embedding)
    
    def embed_with_openai_mock(self, question: str) -> np.ndarray:
        # Mock the OpenAI API for testing purposes
        embedding = np.random.rand(self.vector_dim)
        return self._ensure_fixed_length(embedding)

    def _ensure_fixed_length(self, embedding: np.ndarray) -> np.ndarray:
        # Ensure the embedding is of fixed length
        if len(embedding) != self.vector_dim:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match the fixed dimension {self.vector_dim}.")
        return embedding
    
    def calc_api_cost(self, tokens: int, model: str) -> float:
        if model == "text-embedding-3-small": # $0.020 / 1M tokens
            return tokens * 0.020 / 1000000
        elif model == "text-embedding-3-large": # $0.130 / 1M tokens
            return tokens * 0.130 / 1000000
        else:
            raise ValueError(f"Unsupported model: {model}")

    def embed_schema_with_relevance(self, schema_data: dict, embed_method: str = 'mock') -> dict:
        """
        Embed tables and columns from schema data and include relevance information.
        
        Args:
            schema_data: Dictionary containing database schema with relevance information
            embed_method: String specifying embedding method ('mock', 'small', 'large')
            
        Returns:
            Dictionary with embedded schema elements and their relevance
        """
        embedding_methods = {
            'api_mock': self.embed_with_openai_mock,
            'api_small': self.embed_with_openai_small,
            'api_large': self.embed_with_openai_large,
            'sentence_transformer': self.embed_with_sentence_transformer,
            'bert': self.embed_with_bert
        }
        
        if embed_method not in embedding_methods:
            raise ValueError(f"[! Error] Unsupported embedding method: {embed_method}. "
                            f"Choose from {list(embedding_methods.keys())}")
        
        embed_func = embedding_methods[embed_method]
        
        embedded_schema = {
            'database': schema_data['database'],
            'question': schema_data['question'],
            'id': schema_data.get('id'),
            'tables': [],
            'remarks': schema_data.get('remarks', '')
        }
        
        try:
            embedded_schema['question_embedding'] = embed_func(schema_data['question']).tolist()
            
            for table in schema_data['tables']:
                embedded_table = {
                    'name': table['name'],
                    'relevant': table['relevant'],
                    'embedding': embed_func(table['name']).tolist(),
                    'columns': []
                }
                
                for column in table['columns']:
                    embedded_column = {
                        'name': column['name'],
                        'relevant': column['relevant'],
                        'embedding': embed_func(f"{table['name']}.{column['name']}").tolist()
                    }
                    embedded_table['columns'].append(embedded_column)
                
                embedded_schema['tables'].append(embedded_table)
            
        except Exception as e:
            embedded_schema['remarks'] += f"\nEmbedding error ({embed_method}): {str(e)}"
        
        return embedded_schema


# Example usage
# if __name__ == "__main__":

    # api_key=os.getenv("OPENAI_API_KEY")
    # base_url=os.getenv("OPENAI_BASE_URL")

    # embedder = NLEmbedder(openai_api_key=api_key, base_url=base_url)
    # question = "What is the capital of France?"

    # Get embeddings from different methods
    # vector_sentence_transformer = embedder.embed_with_sentence_transformer(question)
    # print("Sentence Transformer Vector:", vector_sentence_transformer)

    # vector_bert = embedder.embed_with_bert(question)
    # print("BERT Vector:", vector_bert)

    # vector_openai_small = embedder.embed_with_openai_small(question)
    # print("OpenAI API Small Vector:", vector_openai_small)

    # vector_openai_large = embedder.embed_with_openai_large(question)
    # print("OpenAI API Large Vector:", vector_openai_large)

    # vector_openai_mock = embedder.embed_with_openai_mock(question)
    # print("OpenAI API Mock Vector:", vector_openai_mock)
