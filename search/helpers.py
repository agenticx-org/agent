from model2vec import StaticModel
import pandas as pd
import os
import pickle
from umap import UMAP
from hdbscan import HDBSCAN
#import fast_hdbscan
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
from sklearn.feature_extraction.text import CountVectorizer
import spacy
spacy.cli.download("en_core_web_sm")

# Saving data to a pickle file
def save_to_pickle(data, file_path):
    """
    Save data to a pickle file.
    
    Args:
        data: The data to save (can be almost any Python object)
        file_path (str): Path where the pickle file will be saved
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data successfully saved to {file_path}")

# Loading data from a pickle file
def load_from_pickle(file_path):
    """
    Load data from a pickle file.
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        The unpickled data
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def check_file_exists(file_path):
    """
    Check if a file exists at the specified path.
    
    Args:
        file_path (str): The path to the file to check.
        
    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)


embedding_model = StaticModel.from_pretrained("minishlab/M2V_base_output")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
#hdbscan_model = fast_hdbscan.HDBSCAN(min_cluster_size=15)


vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# All representation models

# KeyBERT
keybert_model = KeyBERTInspired()

# Part-of-Speech
pos_model = PartOfSpeech("en_core_web_sm")

# MMR
mmr_model = MaximalMarginalRelevance(diversity=0.3)

representation_model = {
    "KeyBERT": keybert_model,
    "MMR": mmr_model,
    "POS": pos_model
}