# Information retrieval

* Semantic Search: The model ranks and retrieves abstracts by the cosine similarity between the embedding of query and abstracts(documents). At the meanwhile, the project contains a simplified TF-IDF ranking algorithm as the baseline.
* Pretrained model: This project is based on Sentence-Transformers (sbert). The model I use is a relatively outdated one, 'distilbert-base-nli-mean-tokens'.
* Fine-tuning: A siamese network structure is used in training, which computes the embedding from a pair of sentences and their similarity. The objective is to match the title and abstract. I assume there are close semantic relationship between a paper's title and abstract, which is still uncertain.
* Evaluation: Since the project is still in its primary form, there is only a simple top-1 accuracy comparison between TF-IDF and sbert model. Currently, the sbert model still performs worse than tf-idf baseline.
  * TF-IDF accuracy: 0.6346 sbert accuracy: 0.6208