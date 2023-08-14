from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import numpy as np
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
import os 

class SimilarityModel:
    def __init__(self, device, model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model, device=device)
        if 'gpt' in model:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        print("***************** Device:", device)


    def get_similarity(self, a, b):
        a = self.model.encode([a], show_progress_bar=False)[0]
        b = self.model.encode([b], show_progress_bar=False)[0]
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class ClusteringModel:
    def __init__(self, device, model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model, device=device)
        if 'gpt' in model:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

    def cluster_examples(self, examples, k, dissimilar_together=False):
        '''
        examples: list of strings 
        k: number of clusters
        '''

        if not dissimilar_together:
            model = KMeans(n_clusters=k, random_state=42)
            vectors = self.model.encode(examples, show_progress_bar=False)
            model.fit(vectors)
            return model.labels_

        else:
            model = KMeansConstrained(n_clusters=len(examples) // k, size_min=k, size_max=k, random_state=42)
            vectors = self.model.encode(examples, show_progress_bar=False)
            model.fit(vectors)
            
            ## pick an example from each cluster to form the new clutsers 
            clutser_id_to_example_id = {}
            for i, label in enumerate(model.labels_):
                if label not in clutser_id_to_example_id:
                    clutser_id_to_example_id[label] = []
                clutser_id_to_example_id[label].append(i)
            
            new_labels = [0] * len(examples)
            
            for _, example_ids in clutser_id_to_example_id.items():
                for i, example_id in enumerate(example_ids):
                    new_labels[example_id] = i
            return new_labels

if __name__ == "__main__":
    model = SimilarityModel(model='gpt2-medium', device="cuda:1")
    print(model.get_similarity("He is such a great man", "He is beautiful"))