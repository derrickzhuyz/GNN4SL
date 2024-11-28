import os
import torch
import numpy as np
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


class NLEmbedder:
    def __init__(self, vector_dim=384, openai_api_key=None, base_url=None):
        # Load the local SentenceTransformer model
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
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


    """
    NOTE: By default, the length of the embedding vector will be 384 for all-MiniLM-L6-v2 before normalization
    Generate the embedding using SentenceTransformer
    :param nl: str, the natural language to embed
    :return: np.ndarray, the embedding of the natural language
    """
    def embed_with_sentence_transformer(self, nl: str) -> np.ndarray:
        embedding = self.sentence_model.encode(nl)
        return self._normalize_embedding_dimension(embedding)


    """
    NOTE: By default, the length of the embedding vector will be 768 for bert-base-uncased before normalization
    Generate the embedding using BERT
    :param nl: str, the natural language to embed
    :return: np.ndarray, the embedding of the natural language
    """
    def embed_with_bert(self, nl: str) -> np.ndarray:
        # Tokenize the input question
        inputs = self.bert_tokenizer(nl, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Get the embeddings from BERT
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # Get the mean of the last hidden state as the sentence embedding
        last_hidden_state = outputs.last_hidden_state
        sentence_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
        return self._normalize_embedding_dimension(sentence_embedding)


    """
    NOTE: By default, the length of the embedding vector will be 1536 for text-embedding-3-small or 3072 for text-embedding-3-large before normalization
    Generate the embedding using OpenAI API (small, large embedding model, and mock for testing)
    :param nl: str, the natural language to embed
    :return: np.ndarray, the embedding of the natural language
    """
    def embed_with_openai_small(self, nl: str) -> np.ndarray:
        response = self.client.embeddings.create(input=nl, model="text-embedding-3-small")
        embedding = np.array(response.data[0].embedding)
        print('cost:', response.usage.total_tokens * 0.020 / 1000000) # $0.020 / 1M tokens
        return self._normalize_embedding_dimension(embedding)
    
    def embed_with_openai_large(self, nl: str) -> np.ndarray:
        response = self.client.embeddings.create(input=nl, model="text-embedding-3-large")
        embedding = np.array(response.data[0].embedding)
        print('cost:', response.usage.total_tokens * 0.130 / 1000000) # $0.130 / 1M tokens
        return self._normalize_embedding_dimension(embedding)
    
    def embed_with_openai_mock(self, nl: str) -> np.ndarray:
        # Mock the OpenAI API for testing purposes
        embedding = np.random.rand(self.vector_dim)
        return self._normalize_embedding_dimension(embedding)


    """
    Normalize embedding to fixed length by either truncating, padding, or interpolating
    :param embedding: np.ndarray, the original embedding vector
    :return: np.ndarray, the normalized embedding with length self.vector_dim
    """
    def _normalize_embedding_dimension(self, embedding: np.ndarray) -> np.ndarray:
        orig_dim = len(embedding)
        
        # If dimensions match, return original
        if orig_dim == self.vector_dim:
            return embedding
        
        # If original is longer, use interpolation to reduce dimension
        if orig_dim > self.vector_dim:
            indices = np.linspace(0, orig_dim-1, self.vector_dim, dtype=int)
            return embedding[indices]
        
        # If original is shorter, use padding with zeros
        if orig_dim < self.vector_dim:
            padded = np.zeros(self.vector_dim)
            padded[:orig_dim] = embedding
            return padded


    """
    Calculate the cost of the OpenAI API
    :param tokens: int, the number of tokens
    :param model: str, the model used
    :return: float, the cost of the API
    """
    def calc_api_cost(self, tokens: int, model: str) -> float:
        if model == "text-embedding-3-small": # $0.020 / 1M tokens
            return tokens * 0.020 / 1000000
        elif model == "text-embedding-3-large": # $0.130 / 1M tokens
            return tokens * 0.130 / 1000000
        else:
            raise ValueError(f"Unsupported model: {model}")


    """
    NOTE: Deprecated! This function is deprecated for lots of embedding of names in databases will be stored repeatedly. 
        Now we use the function embed_schema_element_names() to embed the schema data.
    Embed the schema data: table names, column names, and question
    :param schema_data: dict, the schema data to embed
    :param embed_method: str, the embedding method to use
    :return: dict, the embedded schema data
    """
    def _deprecated_embed_name_and_question(self, schema_data: dict, embed_method: str = 'api_mock') -> dict:
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
    

    """
    Embed only the question from schema data
    :param schema_data: dict, the schema data to embed
    :param embed_method: str, the embedding method to use
    :return: dict, the embedded schema data
    """
    def embed_question(self, schema_data: dict, embed_method: str = 'api_mock') -> dict:
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
        embedded_schema = schema_data.copy()
        
        try:
            embedded_schema['question_embedding'] = embed_func(schema_data['question']).tolist()
        except Exception as e:
            embedded_schema['remarks'] += f"\nEmbedding error ({embed_method}): {str(e)}"
        
        return embedded_schema


    """
    NOTE: Now we use this function to embed the schema data: database name, table names, and column names
    Embed database schema: database name, table names, and column names
    :param schema_data: dict, the database schema to embed
    :param embed_method: str, the embedding method to use
    :return: dict, the embedded schema data
    """
    def embed_schema_element_names(self, schema_data: dict, embed_method: str = 'api_mock') -> dict:
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
        else:
            print(f"[INFO] {embed_method} is used for embedding method.")

        embed_func = embedding_methods[embed_method]
        
        # Create a deep copy of the original schema
        embedded_schema = schema_data.copy()
        
        try:
            # Add database name embedding
            embedded_schema['database_name_embedding'] = embed_func(schema_data['database']).tolist()
            
            # Add embeddings for tables and columns while preserving original structure
            for table in embedded_schema['tables']:
                # Add table name embedding
                table['table_name_embedding'] = embed_func(table['table']).tolist()
                
                # Add column embeddings
                table['column_name_embeddings'] = {}
                for column in table['columns']:
                    # Create context-aware column name by combining table and column
                    column_context = f"{table['table']}.{column}"
                    table['column_name_embeddings'][column] = embed_func(column_context).tolist()
                
        except Exception as e:
            if 'remarks' not in embedded_schema:
                embedded_schema['remarks'] = ''
            embedded_schema['remarks'] += f"\nEmbedding error ({embed_method}): {str(e)}"
        
        return embedded_schema


# Example usage
if __name__ == "__main__":
    # api_key=os.getenv("OPENAI_API_KEY")
    # base_url=os.getenv("OPENAI_BASE_URL")

    # embedder = NLEmbedder(openai_api_key=api_key, base_url=base_url)
    # question = "How many singers do we have?"

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

    pass
