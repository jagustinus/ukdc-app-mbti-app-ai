import pandas as pd
import numpy as np
import re
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from io import StringIO

class MBTIJobPredictor:
    def __init__(self, csv_data):
        """
        Initialize the predictor with CSV data containing MBTI and profession info

        Args:
            csv_data (str): CSV string containing the MBTI and job data
        """
        # Parse the CSV data into a DataFrame
        self.df = pd.read_csv(StringIO(csv_data))

        # Create mappings and train the model
        self.mbti_types = sorted(self.df['mbti'].unique())
        self.job_categories = sorted(self.df['field'].str.cat(self.df['subcategory'], sep=' - ').unique())
        self.process_data()

    def process_data(self):
        """Process the data and train a KNN model"""
        # Create MBTI features (one-hot encoded)
        self.mbti_encoder = OneHotEncoder(sparse_output=False)
        self.mbti_encoder.fit(np.array(self.mbti_types).reshape(-1, 1))

        # Create job labels
        self.df['job_category'] = self.df['field'] + ' - ' + self.df['subcategory']

        # Map MBTI to jobs for the basic approach
        self.mbti_to_jobs = {}
        for mbti in self.mbti_types:
            jobs = self.df[self.df['mbti'] == mbti]['job_category'].tolist()
            self.mbti_to_jobs[mbti] = Counter(jobs)

        # Prepare data for KNN model
        X = self.mbti_encoder.transform(self.df['mbti'].values.reshape(-1, 1))
        y = self.df['job_category'].values

        # Train KNN model
        self.knn_model = KNeighborsClassifier(n_neighbors=1)
        self.knn_model.fit(X, y)
        joblib.dump(self.knn_model, "model.pkl")

    def parse_mbti_input(self, mbti_input):
        """
        Parse the input string of MBTI types with probabilities

        Args:
            mbti_input (str): String like "ESFJ 0.2 ISTJ 0.3"

        Returns:
            dict: Dictionary of MBTI types to probabilities
        """
        mbti_probabilities = {}
        pattern = r'([A-Z]{4})\s+(0\.\d+|\d+\.\d+)'
        matches = re.findall(pattern, mbti_input)

        for mbti, prob in matches:
            mbti_probabilities[mbti] = float(prob)

        return mbti_probabilities

    def predict_jobs(self, mbti_input, top_n=5):
        """
        Predict suitable jobs based on a weighted combination of MBTI types

        Args:
            mbti_input (str): String like "ESFJ 0.2 ISTJ 0.3"
            top_n (int): Number of top job recommendations to return

        Returns:
            list: List of predicted job categories with their scores
        """
        mbti_probabilities = self.parse_mbti_input(mbti_input)

        # Method 1: Basic weighted counting approach
        job_scores = Counter()

        for mbti, probability in mbti_probabilities.items():
            if mbti in self.mbti_to_jobs:
                for job, count in self.mbti_to_jobs[mbti].items():
                    job_scores[job] += probability * count

        # Method 2: KNN approach
        # Create a weighted feature vector
        feature_vector = np.zeros((1, len(self.mbti_types)))

        for mbti, probability in mbti_probabilities.items():
            if mbti in self.mbti_types:
                idx = self.mbti_types.index(mbti)
                feature_vector[0, idx] = probability

        # Normalize to sum to 1 (if not already)
        if np.sum(feature_vector) > 0:
            feature_vector = feature_vector / np.sum(feature_vector)

        # Get nearest neighbors with distances
        distances, indices = self.knn_model.kneighbors(feature_vector, n_neighbors=min(5, len(self.df)))

        # Add KNN results to job scores with distance-based weighting
        for i, idx in enumerate(indices[0]):
            job = self.df['job_category'].iloc[idx]
            # Convert distance to similarity score (1 / (1 + distance))
            similarity = 1 / (1 + distances[0][i])
            job_scores[job] += similarity

        # Return top N job recommendations
        return job_scores.most_common(top_n)

    def add_data(self, name, field, subcategory, mbti):
        """
        Add new data to the predictor and retrain the model

        Args:
            name (str): Person's name
            field (str): Field of work
            subcategory (str): Job subcategory
            mbti (str): MBTI personality type
        """
        new_row = pd.DataFrame({
            'name': [name],
            'field': [field],
            'subcategory': [subcategory],
            'mbti': [mbti]
        })

        self.df = pd.concat([self.df, new_row], ignore_index=True)

        # Update MBTI types and job categories if new ones are added
        if mbti not in self.mbti_types:
            self.mbti_types.append(mbti)
            self.mbti_types.sort()

        job_category = f"{field} - {subcategory}"
        if job_category not in self.job_categories:
            self.job_categories.append(job_category)
            self.job_categories.sort()

        # Retrain the model with updated data
        self.process_data()


# Example usage
if __name__ == "__main__":
    with open("./data/raw_mbti.csv", 'r', encoding='utf-8') as f:
        csv_data = f.read()
        predictor = MBTIJobPredictor(csv_data)

        mbti_input = "INTP 0.7 ENFP 0.6"
        predictions = predictor.predict_jobs(mbti_input)

        print(f"Based on MBTI distribution: {mbti_input}")
        print("\nRecommended jobs:")
        for i, (job, score) in enumerate(predictions, 1):
            print(f"{i}. {job} (Score: {score:.2f})")
