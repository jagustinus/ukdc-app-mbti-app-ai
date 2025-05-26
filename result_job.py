import pandas as pd
import numpy as np
import re
import joblib
from io import StringIO
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import random

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
        self.df['job_category'] = self.df['subcategory']  # Simplified to just use subcategory

        # Map MBTI to jobs for the basic approach
        self.mbti_to_jobs = {}
        # Also track category distribution for diversity enhancement
        self.job_category_counts = Counter(self.df['job_category'])

        for mbti in self.mbti_types:
            jobs = self.df[self.df['mbti'] == mbti]['job_category'].tolist()
            self.mbti_to_jobs[mbti] = Counter(jobs)

        # Prepare data for the Decision Tree model
        X = self.mbti_encoder.transform(self.df['mbti'].values.reshape(-1, 1))
        y = self.df['job_category'].values

        # Train Decision Tree model with reduced max_depth to avoid overfitting
        self.dt_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2)
        self.dt_model.fit(X, y)

        joblib.dump(self.dt_model, "model.pkl")

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

    def predict_jobs(self, mbti_input, top_n=3, diversity_factor=0.3):
        """
        Predict suitable jobs based on a weighted combination of MBTI types

        Args:
            mbti_input (str or dict): String like "ESFJ 0.2 ISTJ 0.3" or dictionary
            top_n (int): Number of top job recommendations to return
            diversity_factor (float): Factor to control diversity (0-1, higher means more diverse)

        Returns:
            list: List of predicted job categories with their scores
        """
        mbti_probabilities = self.parse_mbti_input(mbti_input)

        # Method 1: Basic weighted counting approach (40% weight)
        counting_scores = Counter()

        for mbti, probability in mbti_probabilities.items():
            if mbti in self.mbti_to_jobs:
                for job, count in self.mbti_to_jobs[mbti].items():
                    # Apply inverse frequency weighting to boost rare categories
                    category_frequency = self.job_category_counts[job] / sum(self.job_category_counts.values())
                    diversity_boost = 1 / (category_frequency + 0.1)  # Avoid division by zero

                    # Apply the diversity boost with controllable strength
                    adjusted_count = count * (1 + diversity_factor * (diversity_boost - 1))
                    counting_scores[job] += probability * adjusted_count

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

                # Add weighted scores with diversity adjustment
                for i, job in enumerate(classes):
                    # Apply diversity boost based on category frequency
                    if job in self.job_category_counts:
                        category_frequency = self.job_category_counts[job] / sum(self.job_category_counts.values())
                        diversity_boost = 1 / (category_frequency + 0.1)
                        adjusted_proba = proba[i] * (1 + diversity_factor * (diversity_boost - 1))
                        dt_scores[job] += adjusted_proba * probability

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

        # Add controlled randomness to break ties and increase diversity
        randomized_scores = {}
        for job, score in combined_scores.items():
            # Add a small random factor to break potential ties and increase diversity
            random_factor = random.uniform(0, 0.15)  # Random boost between 0-15%
            randomized_scores[job] = score * (1 + random_factor)

        # Get top N with randomness applied
        top_jobs = sorted(randomized_scores.items(), key=lambda x: x[1], reverse=True)[:int(top_n*1.5)]

        # Mix in some variety by ensuring at least one less common job category
        final_selection = []

        # Select top jobs while ensuring some diversity
        if len(top_jobs) > top_n:
            # Always include the top job
            final_selection.append(top_jobs[0])

            # Divide remaining jobs into high and lower scoring groups
            high_scoring = top_jobs[1:int(top_n/2)+1]
            lower_scoring = top_jobs[int(top_n/2)+1:]

            # Add some from high scoring
            final_selection.extend(high_scoring[:int(top_n/2)])

            # Add some from lower scoring to increase diversity
            if lower_scoring and len(final_selection) < top_n:
                # Randomly select from lower scoring group
                random_selections = random.sample(lower_scoring, min(top_n - len(final_selection), len(lower_scoring)))
                final_selection.extend(random_selections)

            # Fill any remaining slots with top scoring
            if len(final_selection) < top_n:
                remaining = [j for j in top_jobs if j not in final_selection]
                final_selection.extend(remaining[:top_n - len(final_selection)])
        else:
            final_selection = top_jobs

        # Scale scores to 0-100 range for better readability
        final_scores = []
        for job, score in final_selection[:top_n]:
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

        job_category = subcategory
        if job_category not in self.job_categories:
            self.job_categories.append(job_category)
            self.job_categories.sort()

        # Retrain the model
        self.process_data()
