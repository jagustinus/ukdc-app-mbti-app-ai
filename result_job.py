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
        # Parse the CSV data into a DataFrame
        self.df = pd.read_csv(StringIO(csv_data))

        # Create mappings and train the model
        self.mbti_types = sorted(self.df['mbti'].unique())
        self.job_categories = sorted(self.df['subcategory'].unique())
        self.process_data()

        # Create a sample pool of job categories for randomization
        self._create_category_pools()

    def _create_category_pools(self):
        """Create pools of job categories based on frequency to enforce diversity"""
        # Count job category occurrences
        self.category_counts = Counter(self.df['subcategory'])

        # Calculate quartiles for job frequency
        values = list(self.category_counts.values())

        # Create category pools based on frequency
        self.common_jobs = []
        self.uncommon_jobs = []
        self.rare_jobs = []

        for category, count in self.category_counts.items():
            if count >= np.percentile(values, 75):
                self.common_jobs.append(category)
            elif count >= np.percentile(values, 25):
                self.uncommon_jobs.append(category)
            else:
                self.rare_jobs.append(category)

        # Ensure we have items in each pool
        if not self.rare_jobs and self.uncommon_jobs:
            # Move some from uncommon to rare
            move_count = max(1, len(self.uncommon_jobs) // 3)
            self.rare_jobs = self.uncommon_jobs[:move_count]
            self.uncommon_jobs = self.uncommon_jobs[move_count:]

        if not self.uncommon_jobs and self.common_jobs:
            # Move some from common to uncommon
            move_count = max(1, len(self.common_jobs) // 3)
            self.uncommon_jobs = self.common_jobs[:move_count]
            self.common_jobs = self.common_jobs[move_count:]

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

        # Train Decision Tree model with reduced max_depth and increased min_samples_leaf
        # This helps prevent overfitting to common categories
        self.dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=5)
        self.dt_model.fit(X, y)

        joblib.dump(self.dt_model, "model.pkl")

    def parse_mbti_input(self, mbti_input):
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

    def predict_jobs(self, mbti_input, top_n=3, diversity_factor=0.9, force_diversity=True):
        """
        Predict suitable jobs with enforced diversity

        Args:
            mbti_input (str or dict): String like "ESFJ 0.2 ISTJ 0.3" or dictionary
            top_n (int): Number of top job recommendations to return
            diversity_factor (float): Factor to control diversity (0-1, higher means more diverse)
            force_diversity (bool): If True, forces inclusion of uncommon categories

        Returns:
            list: List of predicted job categories with their scores
        """
        mbti_probabilities = self.parse_mbti_input(mbti_input)

        # Get all potential job categories
        all_job_scores = self._calculate_initial_scores(mbti_probabilities, diversity_factor)

        # Now implement a stratified selection approach to ensure diversity
        if force_diversity and top_n >= 3:
            return self._forced_diverse_selection(all_job_scores, top_n)
        else:
            # Apply randomness to all scores
            randomized_scores = {}
            for job, score in all_job_scores.items():
                # Add a significant random factor to break potential ties and increase diversity
                random_factor = random.uniform(0, 0.3)  # Random boost between 0-30%
                randomized_scores[job] = score * (1 + random_factor)

            # Get top N recommendations
            final_scores = []
            for job, score in sorted(randomized_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
                scaled_score = score * 100  # Scale to 0-100
                final_scores.append((job, scaled_score))

            return final_scores

    def _calculate_initial_scores(self, mbti_probabilities, diversity_factor):
        """Calculate initial scores for all job categories"""
        # Method 1: Basic weighted counting approach (40% weight)
        counting_scores = Counter()

        # Get total counts for frequency calculations
        total_job_counts = sum(self.category_counts.values())

        for mbti, probability in mbti_probabilities.items():
            if mbti in self.mbti_to_jobs:
                for job, count in self.mbti_to_jobs[mbti].items():
                    # Apply stronger inverse frequency weighting
                    category_frequency = self.category_counts[job] / total_job_counts
                    diversity_boost = 1 / (category_frequency ** 2 + 0.01)  # Square to amplify effect
                    adjusted_count = count * (1 + diversity_factor * diversity_boost)
                    counting_scores[job] += probability * adjusted_count

        # Method 2: Decision Tree approach (60% weight)
        dt_scores = Counter()

        for mbti, probability in mbti_probabilities.items():
            if mbti in self.mbti_types:
                # Create one-hot encoded feature
                idx = self.mbti_types.index(mbti)
                feature_vector = np.zeros((1, len(self.mbti_types)))
                feature_vector[0, idx] = 1

                # Get prediction probabilities
                proba = self.dt_model.predict_proba(feature_vector)[0]
                classes = self.dt_model.classes_

                # Add weighted scores with stronger diversity adjustment
                for i, job in enumerate(classes):
                    if job in self.category_counts:
                        category_frequency = self.category_counts[job] / total_job_counts
                        diversity_boost = 1 / (category_frequency ** 2 + 0.01)
                        adjusted_proba = proba[i] * (1 + diversity_factor * diversity_boost)
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

        return combined_scores

    def _forced_diverse_selection(self, all_job_scores, top_n):
        """Ensure diversity by forcing selections from different frequency pools"""
        final_selection = []

        # Sort all jobs by score
        sorted_jobs = sorted(all_job_scores.items(), key=lambda x: x[1], reverse=True)

        # Determine how many to select from each category
        # With more aggressive diversity distribution
        if top_n >= 5:
            # For larger recommendation sets, include more rare jobs
            common_count = top_n // 3
            uncommon_count = top_n // 3
            rare_count = top_n - common_count - uncommon_count
        elif top_n >= 3:
            common_count = 1
            uncommon_count = 1
            rare_count = top_n - 2
        else:
            # For very small recommendation sets, use a mixed approach
            common_count = 1 if top_n > 1 else 0
            uncommon_count = 0
            rare_count = top_n - common_count

        # Function to get top N from a specific category pool
        def get_top_from_pool(pool, n):
            pool_jobs = []
            for job, score in sorted_jobs:
                if job in pool:
                    pool_jobs.append((job, score))
                    if len(pool_jobs) >= n:
                        break
            return pool_jobs

        # If not enough jobs in a pool, adjust the counts
        if len(self.common_jobs) < common_count:
            uncommon_count += (common_count - len(self.common_jobs))
            common_count = len(self.common_jobs)

        if len(self.uncommon_jobs) < uncommon_count:
            rare_count += (uncommon_count - len(self.uncommon_jobs))
            uncommon_count = len(self.uncommon_jobs)

        if len(self.rare_jobs) < rare_count:
            common_count += (rare_count - len(self.rare_jobs))
            rare_count = len(self.rare_jobs)

        # Get jobs from each pool
        common_selections = get_top_from_pool(self.common_jobs, common_count)
        uncommon_selections = get_top_from_pool(self.uncommon_jobs, uncommon_count)
        rare_selections = get_top_from_pool(self.rare_jobs, rare_count)

        # Combine all selections
        all_selections = []
        all_selections.extend(common_selections)
        all_selections.extend(uncommon_selections)
        all_selections.extend(rare_selections)

        # If still not enough, add more from top scoring jobs
        if len(all_selections) < top_n:
            remaining_needed = top_n - len(all_selections)
            selected_jobs = [job for job, _ in all_selections]

            for job, score in sorted_jobs:
                if job not in selected_jobs:
                    all_selections.append((job, score))
                    remaining_needed -= 1
                    if remaining_needed <= 0:
                        break

        # Add some randomness to the order
        random.shuffle(all_selections)

        # Scale scores to 0-100 range for better readability
        final_scores = []
        for job, score in all_selections[:top_n]:
            # Add small random variation to scores
            scaled_score = score * 100 * random.uniform(0.85, 1.15)
            final_scores.append((job, scaled_score))

        # Sort by score for the final result
        return sorted(final_scores, key=lambda x: x[1], reverse=True)

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

        # Retrain the model and update category pools
        self.process_data()
        self._create_category_pools()
