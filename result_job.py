import pandas as pd
import numpy as np
import re
import joblib
from io import StringIO
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

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
        self.job_categories = sorted(self.df['subcategory'].unique())
        self.process_data()

    def process_data(self):
        """Process the data and train a Decision Tree model"""
        # Create MBTI features (one-hot encoded)
        self.mbti_encoder = OneHotEncoder(sparse_output=False)
        self.mbti_encoder.fit(np.array(self.mbti_types).reshape(-1, 1))

        # Create job labels
        self.df['job_category'] = self.df['subcategory']

        # Map MBTI to jobs for the basic approach
        self.mbti_to_jobs = {}
        for mbti in self.mbti_types:
            jobs = self.df[self.df['mbti'] == mbti]['job_category'].tolist()
            self.mbti_to_jobs[mbti] = Counter(jobs)

        # Prepare data for the Decision Tree model
        X = self.mbti_encoder.transform(self.df['mbti'].values.reshape(-1, 1))
        y = self.df['job_category'].values

        # Train Decision Tree model
        self.dt_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)
        self.dt_model.fit(X, y)

        joblib.dump(self.dt_model, "model.pkl")
        # self.dt_model = joblib.load("model.pkl")

        # Get feature importances for later use
        self.feature_importances = self.dt_model.feature_importances_

    def parse_mbti_input(self, mbti_input):
        """
        Parse the input string of MBTI types with probabilities

        Args:
            mbti_input (str or dict): String like "ESFJ 0.2 ISTJ 0.3" or dict of MBTI types

        Returns:
            dict: Dictionary of MBTI types to probabilities
        """
        if isinstance(mbti_input, str):
            mbti_probabilities = {}
            pattern = r'([A-Z]{4})\s+(0\.\d+|\d+\.\d+)'
            matches = re.findall(pattern, mbti_input)

            for mbti, prob in matches:
                mbti_probabilities[mbti] = float(prob)

            return mbti_probabilities
        else:
            # If it's already a dictionary, just return it
            return mbti_input

    def predict_jobs(self, mbti_input, top_n=3):
        """
        Predict suitable jobs based on a weighted combination of MBTI types

        Args:
            mbti_input (str or dict): String like "ESFJ 0.2 ISTJ 0.3" or dictionary
            top_n (int): Number of top job recommendations to return

        Returns:
            list: List of predicted job categories with their scores
        """
        mbti_probabilities = self.parse_mbti_input(mbti_input)

        # Method 1: Basic weighted counting approach (40% weight)
        counting_scores = Counter()

        for mbti, probability in mbti_probabilities.items():
            if mbti in self.mbti_to_jobs:
                for job, count in self.mbti_to_jobs[mbti].items():
                    counting_scores[job] += probability * count

        # Method 2: Decision Tree approach (60% weight)
        dt_scores = Counter()

        # Create feature vectors for each input MBTI type
        for mbti, probability in mbti_probabilities.items():
            if mbti in self.mbti_types:
                # Create one-hot encoded feature
                idx = self.mbti_types.index(mbti)
                feature_vector = np.zeros((1, len(self.mbti_types)))
                feature_vector[0, idx] = 1

                # Get prediction probabilities for all classes
                proba = self.dt_model.predict_proba(feature_vector)[0]
                classes = self.dt_model.classes_

                # Add weighted scores
                for i, job in enumerate(classes):
                    dt_scores[job] += proba[i] * probability

        # Combine scores with weights
        combined_scores = Counter()

        # Normalize each set of scores
        if counting_scores:
            max_counting = max(counting_scores.values())
            for job, score in counting_scores.items():
                normalized_score = score / max_counting if max_counting > 0 else 0
                combined_scores[job] += 0.4 * normalized_score

        if dt_scores:
            max_dt = max(dt_scores.values())
            for job, score in dt_scores.items():
                normalized_score = score / max_dt if max_dt > 0 else 0
                combined_scores[job] += 0.6 * normalized_score

        # Scale scores to 0-100 range for better readability
        final_scores = []
        for job, score in combined_scores.most_common(top_n):
            scaled_score = score * 100  # Scale to 0-100
            final_scores.append((job, scaled_score))

        return final_scores

    def add_data(self, name, subcategory, mbti):
        """Add new data and retrain the model"""
        new_row = pd.DataFrame({
            'name': [name],
            'subcategory': [subcategory],
            'mbti': [mbti]
        })

        self.df = pd.concat([self.df, new_row], ignore_index=True)

        # Update types if needed
        if mbti not in self.mbti_types:
            self.mbti_types.append(mbti)
            self.mbti_types.sort()

        # job_category = f"{field} - {subcategory}"
        job_category = f"{subcategory}"
        if job_category not in self.job_categories:
            self.job_categories.append(job_category)
            self.job_categories.sort()

        # Retrain the model
        self.process_data()
