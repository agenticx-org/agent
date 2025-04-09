from bertopic import BERTopic
from search.helpers import *


df = pd.read_pickle("dump.pkl")
# Create TF-IDF sparse matrix
docs = df['text'].to_list()

if check_file_exists("embeddings.pkl"):
    print("Loading from old dump")
    embeddings = load_from_pickle("embeddings.pkl")

else:
    print("Creating new embeddings")
    embeddings = embedding_model.encode(docs)
    save_to_pickle(embeddings, "embeddings.pkl")

print(len(embeddings))

if check_file_exists("bert_topic.pkl"):
    print("Loading from a previously fit model")

    topic_model = BERTopic.load("bert_topic.pkl")

else:
   
    topic_model = BERTopic(

    # Pipeline models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    calculate_probabilities=False,
    # Hyperparameters
    top_n_words=10,
    verbose=True
    )

    # Train model
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Show topics
    print(topic_model.get_topic_info())

    topic_model.save("bert_topic.pkl", serialization="pickle")