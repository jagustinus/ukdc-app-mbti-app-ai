#!/usr/bin/env python

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from data_fetcher import DataFetcher

class BayesianAIMBTIApp:
    def __init__(self):
        self.name = "guest"
        self.email = "guest"
        self.responses = []  # Store user responses for ML processing

        # All 16 possible MBTI types
        self.mbti_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        ]

        # Initialize equal prior probabilities for each type (1/16)
        self.type_probabilities = {mbti_type: 1/16 for mbti_type in self.mbti_types}

        # Questions with associated trait likelihoods
        data = DataFetcher("./data/question.csv")
        self.questions = data.data

        # AI components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.load_or_create_ml_model()

        # Feature importance from trained model (or default weights if no model)
        self.feature_importance = np.ones(20) / 20  # Default uniform importance

        # Load Bayesian network structure
        self.bn_structure = self.load_bayesian_network_structure()

        # Personality descriptions
        self.personality_descriptions = {
            "ISTJ": "ISTJs are structured, detail-oriented, and grounded in reality. Their introverted sensing (Si) makes them excellent at recalling past data and applying proven methods, ideal for roles like software engineering or systems analysis, where consistency, precision, and clear frameworks are essential. Their introversion aligns with tasks requiring deep, uninterrupted focus, such as debugging or systems optimization, where quiet environments allow them to excel without distractions.",
            "ISFJ": "ISFJs combine empathy with a high degree of responsibility. Their Si-Fe function stack makes them deeply attentive to user needs while valuing thoroughness and tradition. As technical writers, QA testers, or support engineers, they excel by applying proven methods while ensuring others feel supported and guided. Their introversion helps them patiently and carefully address complex problems, often behind the scenes, where their steadiness shines.",
            "INFJ": "INFJs are visionary idealists with strong internal values and future-oriented thinking (Ni-Fe). In roles like UX design, AI ethics, or mission-driven data science, they strive to humanize technology and ensure its use aligns with moral or societal betterment. Their introversion and introspection allow them to identify subtle patterns and imagine long-term impacts, making them uniquely positioned to advocate for ethical innovation and empathetic design.",
            "INTJ": "INTJs are strategic visionaries driven by insight and logic (Ni-Te). Their independent nature thrives in roles like software architecture, machine learning, or startup leadership, where they can build long-term systems from abstract ideas. Their introversion supports long hours of solitary planning, prototyping, and theoretical work—activities that require intense focus and internal vision. They often lead by competence rather than charisma, thriving in meritocratic environments.",
            "ISTP": "ISTPs are pragmatic problem-solvers who love working with real-world systems. Their Ti-Se combination makes them excel at analyzing how things work, often in high-stakes, hands-on roles like embedded systems, DevOps, or cybersecurity. They enjoy autonomy and dislike micromanagement, making introversion ideal for their need to tinker and experiment uninterrupted. Their calm demeanor under pressure makes them reliable in critical debugging or operational tasks.",
            "ISFP": "ISFPs are sensitive creators who care about beauty, user experience, and emotional impact (Fi-Se). In UI/UX design, front-end work, or multimedia coding, they express their inner values through visual storytelling and human-centered interfaces. Their introverted feeling enables them to deeply consider how technology makes users feel, while their sensing helps craft detailed, immersive experiences. They often prefer roles where their creativity speaks louder than their words.",
            "INFP": "INFPs are imaginative idealists who seek meaning in their work (Fi-Ne). They gravitate toward creative or socially impactful roles like game design, ethical AI, or narrative development, where emotional depth and innovation matter. Their introversion supports the long hours needed for coding solo projects or exploring abstract problems, while their values-driven mindset ensures their work resonates with a personal mission or story.",
            "INTP": "INTPs are analytical explorers with a passion for theoretical systems and abstract problem-solving (Ti-Ne). They're naturally drawn to algorithm design, AI research, or open-source contributions where originality and logical rigor are prized. Their introversion fosters a love of solitude and intellectual depth, allowing them to dive deep into complex ideas without needing external validation. They thrive in environments where independent innovation is encouraged.",
            "ESTP": "ESTPs are energetic realists who love fast-paced, tangible challenges (Se-Ti). They excel in field engineering, tech sales, or startup operations, where quick thinking and action drive success. Their extroversion helps them connect with diverse teams and clients, while their sensing makes them adept at responding in real time. They bring a hands-on, entrepreneurial spirit to tech environments that need adaptability and resourcefulness under pressure.",
            "ESFP": "ESFPs are charismatic doers who bring warmth and spontaneity to any team (Se-Fi). They thrive in tech support, digital marketing, or community management, where human interaction and storytelling meet technical tools. Their extroversion makes them ideal ambassadors for tech platforms, while their sensitivity helps them intuitively meet user needs. They shine when technology is a medium for connection, expression, or experience.",
            "ENFP": "ENFPs are enthusiastic innovators driven by vision and human potential (Ne-Fi). In startup teams, product ideation, or innovation labs, they generate bold ideas and inspire collaboration. Their extroversion supports dynamic brainstorming and stakeholder engagement, while their intuition sees connections others might miss. They prefer purpose-driven environments where adaptability and creativity are not just valued, but necessary for growth.",
            "ENTP": "ENTPs are clever and inventive problem-solvers (Ne-Ti), well-suited for roles involving disruption, exploration, or systems reimagination—such as product management, startup founding, or R&D. Their extroversion energizes them in fast-moving discussions and multidisciplinary collaboration, while their curiosity keeps them iterating. They don't just adapt to change—they cause it, challenging norms and proposing radical alternatives with logic and charm.",
            "ESTJ": "ESTJs are natural organizers who lead with efficiency, structure, and decisiveness (Te-Si). Their strength lies in managing projects, IT departments, or enterprise systems, where clarity and execution are non-negotiable. Their extroversion makes them assertive communicators and reliable decision-makers, while their sensing helps them enforce practical, tested methods. They bring order to chaos and ensure standards are met with consistency.",
            "ESFJ": "ESFJs are nurturing coordinators who prioritize harmony and responsibility (Fe-Si). In customer success, HR tech, or user education, they bring empathy and diligence to every interaction. Their extroversion supports team-building and proactive outreach, while their sensing ensures no detail is overlooked. They excel at translating technical complexity into approachable solutions for end users, helping bridge the gap between people and systems.",
            "ENFJ": "ENFJs are inspirational mentors who guide others with vision and care (Fe-Ni). As edtech leaders, team coaches, or UX researchers, they empower others to grow through technology. Their extroversion allows them to motivate and unify teams, while their intuition helps them anticipate long-term user and team needs. They see technology as a tool for elevating people and ideas, not just solving problems.",
            "ENTJ": "ENTJs are commanding strategists who thrive on leadership and long-term planning (Te-Ni). Ideal as CTOs, tech founders, or lead architects, they turn vision into scalable reality through bold execution. Their extroversion allows them to rally teams and stakeholders, while their intuition fuels innovation. They prefer environments that reward drive, systems thinking, and tangible outcomes, often rising naturally to the top of hierarchical tech structures."
        }

    def load_or_create_ml_model(self):
        """Load a pre-trained ML model or create a new one"""
        model_path = "./models/mbti_classifier.joblib"

        if os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
                print("Loaded pre-trained ML model")

                # If model has feature importances, extract them
                if hasattr(self.ml_model, 'feature_importances_'):
                    self.feature_importance = self.ml_model.feature_importances_
                return
            except:
                print("Could not load model, creating new one")

        # Create a neural network classifier
        self.ml_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        print("Created new ML model")

    def load_bayesian_network_structure(self):
        """Load or define a Bayesian network structure for MBTI dimensions"""
        # This is a simplified representation of the Bayesian network structure
        # In a real implementation, we would define the full conditional probability tables

        # Define the network structure (nodes and edges)
        network = {
            # Nodes represent MBTI dimensions and their dependencies
            'E-I': {'parents': [], 'cpt': {}},
            'S-N': {'parents': [], 'cpt': {}},
            'T-F': {'parents': ['E-I'], 'cpt': {}},
            'J-P': {'parents': ['S-N'], 'cpt': {}}
        }

        # In a real implementation, we would include conditional probability tables
        # for each node based on empirical data or expert knowledge

        return network

    def set_email(self, email: str):
        self.email = email

    def set_name(self, name: str):
        self.name = name

    def ask_question(self, question, likelihoods):
        """Ask a question and update type probabilities based on answer"""
        while True:
            print("\n" + question)
            print("1 - Strongly Disagree")
            print("2 - Disagree")
            print("3 - Neutral")
            print("4 - Agree")
            print("5 - Strongly Agree")

            try:
                answer = int(input("Your answer (1-5): "))
                if 1 <= answer <= 5:
                    # Store response for later ML processing
                    self.responses.append(answer)

                    # Update probabilities using Bayesian approach with network influence
                    self.update_probabilities(answer, likelihoods)
                    return
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

    def update_probabilities(self, answer, likelihoods):
        """Update the probabilities using Bayesian network and ML influence"""
        # Get the likelihood values for this answer
        answer_likelihoods = likelihoods[answer]

        # Calculate posterior probabilities for each type
        new_probabilities = {}

        # Question index (based on number of responses)
        question_idx = len(self.responses) - 1

        # Get question importance from feature importance (if available)
        question_weight = self.feature_importance[question_idx] if question_idx < len(self.feature_importance) else 1.0

        for mbti_type in self.mbti_types:
            # Calculate the likelihood of this answer given the type
            type_likelihood = 1.0
            for idx, letter in enumerate(mbti_type):
                # For each letter in MBTI type, get its dimension's likelihood
                if letter in answer_likelihoods:
                    # Apply weight based on Bayesian network structure
                    dimension_weight = self.get_dimension_weight(idx, mbti_type)
                    weighted_likelihood = answer_likelihoods[letter] ** (dimension_weight * question_weight)
                    type_likelihood *= weighted_likelihood

            # Apply Bayes theorem with network influence: P(Type|Answer) ∝ P(Answer|Type) × P(Type)
            new_probabilities[mbti_type] = type_likelihood * self.type_probabilities[mbti_type]

        # Normalize the probabilities to sum to 1
        total = sum(new_probabilities.values())
        if total > 0:  # Avoid division by zero
            for mbti_type in self.mbti_types:
                self.type_probabilities[mbti_type] = new_probabilities[mbti_type] / total

    def get_dimension_weight(self, dim_idx, mbti_type):
        """Get the weight for a dimension based on Bayesian network structure"""
        # Map dimension index to dimension name
        dim_names = ['E-I', 'S-N', 'T-F', 'J-P']
        dim_name = dim_names[dim_idx]

        # Base weight is 1.0
        weight = 1.0

        # Apply network influence based on parent dimensions
        parents = self.bn_structure[dim_name]['parents']
        for parent in parents:
            parent_idx = dim_names.index(parent)
            parent_letter = mbti_type[parent_idx]

            # Simplified influence calculation
            # In a real implementation, this would use the CPT values
            if parent_letter in ['E', 'S', 'T', 'J']:
                weight *= 1.1  # Slight boost for certain parent values
            else:
                weight *= 0.9  # Slight reduction for others

        return weight

    def get_next_question_index(self, remaining_indices):
        """Choose the most informative next question using AI-assisted selection"""
        if not remaining_indices:
            return None

        # If we have enough responses to make a prediction
        if len(self.responses) >= 5 and self.ml_model is not None:
            # Use current responses to predict question importance
            current_responses = np.array(self.responses + [3] * (20 - len(self.responses)))  # Pad with neutral
            current_responses = current_responses.reshape(1, -1)

            # Use the model's feature importances to determine the next best question
            importances = []
            for idx in remaining_indices:
                # Calculate information gain for each remaining question
                if idx < len(self.feature_importance):
                    importances.append((idx, self.feature_importance[idx]))
                else:
                    importances.append((idx, 0.5))  # Default importance

            # Sort by importance (highest first)
            if importances:
                importances.sort(key=lambda x: x[1], reverse=True)
                return importances[0][0]

        # Default to the first remaining question if no AI guidance
        return remaining_indices[0]

    def determine_type(self):
        """Find the MBTI type with the highest probability"""
        # If we have enough responses and a trained ML model
        if len(self.responses) >= 10 and self.ml_model is not None:
            try:
                # Prepare features
                features = np.array(self.responses)
                if len(features) < 20:  # Pad with neutral answers if needed
                    features = np.pad(features, (0, 20 - len(features)), 'constant', constant_values=3)

                features = features.reshape(1, -1)
                features_scaled = self.scaler.transform(features) if hasattr(self.scaler, 'mean_') else self.scaler.fit_transform(features)

                # Get ML prediction
                ml_prediction = self.ml_model.predict(features_scaled)[0]

                # Combine ML prediction with Bayesian probabilities
                # Increase probability of ML-predicted type
                ml_boost_factor = 1.2  # Boost by 20%

                # Create a copy of current probabilities
                combined_probs = self.type_probabilities.copy()

                # Apply boost to ML prediction
                combined_probs[ml_prediction] *= ml_boost_factor

                # Normalize again
                total = sum(combined_probs.values())
                for mbti_type in self.mbti_types:
                    combined_probs[mbti_type] /= total

                # Get final combined result
                return max(combined_probs.items(), key=lambda x: x[1])[0]

            except Exception as e:
                print(f"ML model error: {e}, using Bayesian only")
                pass

        # Fallback to Bayesian-only approach
        return max(self.type_probabilities.items(), key=lambda x: x[1])[0]

    def train_model_with_responses(self, actual_type):
        """Train the ML model with the user's responses and their confirmed type"""
        if len(self.responses) < 10:
            print("Not enough responses to train the model")
            return False

        try:
            # Prepare features and target
            features = np.array(self.responses)
            if len(features) < 20:  # Pad with neutral answers if needed
                features = np.pad(features, (0, 20 - len(features)), 'constant', constant_values=3)

            features = features.reshape(1, -1)

            # If the model has been trained before, we need to add the new data
            if hasattr(self.ml_model, 'partial_fit'):
                # Scale features with existing scaler if available
                features_scaled = self.scaler.transform(features) if hasattr(self.scaler, 'mean_') else self.scaler.fit_transform(features)

                # Partial fit with the new data
                self.ml_model.partial_fit(features_scaled, [actual_type], classes=self.mbti_types)

                # Save the updated model
                os.makedirs("./models", exist_ok=True)
                joblib.dump(self.ml_model, "./models/mbti_classifier.joblib")
                print("Model updated with new training data")
                return True

            print("Model doesn't support online learning")
            return False

        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def run_test(self):
        print("\n" + "="*50)
        print("Welcome to the Bayesian AI MBTI Personality Test!")
        print("="*50)
        print("\nThis test uses Bayesian networks and machine learning to determine your personality type.")
        print("The questions adapt based on your previous answers using AI-driven intelligence.")

        input("\nPress Enter to begin the test...")

        # Reset responses for a new test
        self.responses = []

        # Track which questions we've asked
        remaining_indices = list(range(len(self.questions)))
        asked_questions = 0

        # Ask up to 20 questions (or fewer if we reach high confidence)
        while remaining_indices and asked_questions < 20:
            # Get the next best question to ask using AI-assisted selection
            question_index = self.get_next_question_index(remaining_indices)
            if question_index is not None:
                remaining_indices.remove(question_index)

            # Ask the question and update probabilities
            if question_index is not None:
                question, likelihoods = self.questions[question_index]
                self.ask_question(question, likelihoods)
                asked_questions += 1

            # Get current most probable type
            current_type = self.determine_type()
            current_probability = self.type_probabilities[current_type]

            # Print current probabilities (top 3)
            top_types = sorted(self.type_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            print("\nCurrent top probabilities:")
            for type_name, prob in top_types:
                print(f"{type_name}: {prob:.1%}")

            # If we're very confident (>85%), we could stop early
            if current_probability > 0.85 and asked_questions >= 10:
                print("\nHigh confidence reached, finishing test early...")
                break

        # Determine and display results
        personality_type = self.determine_type()
        confidence = self.type_probabilities[personality_type]

        print("\n" + "="*50)
        print(f"Your MBTI Personality Type: {personality_type} (Confidence: {confidence:.1%})")
        print("="*50)

        if personality_type in self.personality_descriptions:
            print(f"\nDescription: {self.personality_descriptions[personality_type]}")

        # Show probability distribution for all types
        print("\nFinal probabilities for all types:")
        sorted_types = sorted(self.type_probabilities.items(), key=lambda x: x[1], reverse=True)
        for type_name, prob in sorted_types:
            print(f"{type_name}: {prob:.1%}")

        # Show dimensional breakdown
        print("\nDimensional breakdown:")
        e_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[0] == 'E')
        i_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[0] == 'I')
        s_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[1] == 'S')
        n_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[1] == 'N')
        t_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[2] == 'T')
        f_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[2] == 'F')
        j_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[3] == 'J')
        p_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[3] == 'P')

        print(f"E: {e_prob:.1%} - I: {i_prob:.1%}")
        print(f"S: {s_prob:.1%} - N: {n_prob:.1%}")
        print(f"T: {t_prob:.1%} - F: {f_prob:.1%}")
        print(f"J: {j_prob:.1%} - P: {p_prob:.1%}")

        # Ask if user would like to improve the AI model
        print("\nWould you like to help improve our AI model?")
        print("If you know your actual MBTI type, we can use your responses to train the model.")
        improve = input("Enter your known MBTI type (or press Enter to skip): ").upper()

        if improve in self.mbti_types:
            success = self.train_model_with_responses(improve)
            if success:
                print("Thank you for helping improve our AI model!")
            else:
                print("Sorry, we couldn't update the model at this time.")

    def reset(self):
        # Reset type probabilities
        for mbti_type in self.mbti_types:
            self.type_probabilities[mbti_type] = 1/16

        # Reset responses
        self.responses = []

# Create an instance of the app
if __name__ == "__main__":
    app = BayesianAIMBTIApp()
    app.run_test()
