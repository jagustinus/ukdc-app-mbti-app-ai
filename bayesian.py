#!/usr/bin/env python
# old ver by copilot (Claude 3.7)
# TODO: Nentukan Jobs atau cocok
# TODO: Flask

from data_fetcher import DataFetcher

class BayesianMBTIApp:
    def __init__(self):
        # All 16 possible MBTI types
        self.mbti_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        ]

        # Initialize equal prior probabilities for each type (1/16)
        self.type_probabilities = {}
        for i in range(0, len(self.mbti_types)):
            self.type_probabilities[self.mbti_types[i]] = 1/16
        # self.type_probabilities = {mbti_type: 1/16 for mbti_type in self.mbti_types}

        # Questions with associated trait likelihoods
        data = DataFetcher("./data/question.csv")
        self.questions = data.data

        self.personality_descriptions = {
            "ISTJ": "Quiet, serious, thorough, and dependable. Practical, matter-of-fact, realistic, and responsible.",
            "ISFJ": "Quiet, friendly, responsible, and conscientious. Committed and steady in meeting obligations.",
            "INFJ": "Seek meaning and connection in ideas, relationships, and material possessions.",
            "INTJ": "Have original minds and great drive for implementing their ideas and achieving their goals.",
            "ISTP": "Tolerant and flexible, quiet observers until a problem appears, then act quickly to find solutions.",
            "ISFP": "Quiet, friendly, sensitive, and kind. Enjoy the present moment and what's going on around them.",
            "INFP": "Idealistic, loyal to their values and to people who are important to them.",
            "INTP": "Seek to develop logical explanations for everything that interests them.",
            "ESTP": "Flexible and tolerant, take a pragmatic approach focused on immediate results.",
            "ESFP": "Outgoing, friendly, and accepting. Exuberant lovers of life, people, and material comforts.",
            "ENFP": "Warmly enthusiastic and imaginative. See life as full of possibilities.",
            "ENTP": "Quick, ingenious, stimulating, alert, and outspoken. Resourceful in solving challenging problems.",
            "ESTJ": "Practical, realistic, matter-of-fact. Decisive, quickly move to implement decisions.",
            "ESFJ": "Warmhearted, conscientious, and cooperative. Want harmony in their environment.",
            "ENFJ": "Warm, empathetic, responsive, and responsible. Highly attuned to the needs of others.",
            "ENTJ": "Frank, decisive, assumes leadership readily. Quickly see illogical procedures and policies."
        }

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
                    # Update probabilities using Bayes' theorem
                    self.update_probabilities(answer, likelihoods)
                    return
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

    def update_probabilities(self, answer, likelihoods):
        """Update the probabilities of each MBTI type based on the answer"""
        # Get the likelihood values for this answer
        answer_likelihoods = likelihoods[answer]

        # Calculate posterior probabilities for each type
        new_probabilities = {}

        for mbti_type in self.mbti_types:
            # Calculate the likelihood of this answer given the type
            type_likelihood = 1.0
            for _, letter in enumerate(mbti_type):
                # For each letter in MBTI type, get its dimension's likelihood
                if letter in answer_likelihoods:
                    type_likelihood *= answer_likelihoods[letter]

            # Apply Bayes theorem: P(Type|Answer) ∝ P(Answer|Type) × P(Type)
            new_probabilities[mbti_type] = type_likelihood * self.type_probabilities[mbti_type]

        # Normalize the probabilities to sum to 1
        total = sum(new_probabilities.values())
        if total > 0:  # Avoid division by zero
            for mbti_type in self.mbti_types:
                self.type_probabilities[mbti_type] = new_probabilities[mbti_type] / total

    def get_next_question_index(self, remaining_indices):
        """Choose the most informative next question using a simple approach"""
        if not remaining_indices:
            return None

        # Get current most probable type and its closest competitors
        top_types = sorted(self.type_probabilities.items(), key=lambda x: x[1], reverse=True)
        current_leader = top_types[0][0]
        close_competitors = [t[0] for t in top_types[1:4]]  # Get next 3 closest types

        # Find dimensions where there's the most uncertainty
        uncertain_dimensions = []

        # Calculate dimensional probabilities
        e_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[0] == 'E')
        i_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[0] == 'I')

        s_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[1] == 'S')
        n_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[1] == 'N')

        t_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[2] == 'T')
        f_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[2] == 'F')

        j_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[3] == 'J')
        p_prob = sum(self.type_probabilities[t] for t in self.mbti_types if t[3] == 'P')

        # Add dimensions with close probabilities (uncertain dimensions)
        if abs(e_prob - i_prob) < 0.2:
            uncertain_dimensions.append("EI")
        if abs(s_prob - n_prob) < 0.2:
            uncertain_dimensions.append("SN")
        if abs(t_prob - f_prob) < 0.2:
            uncertain_dimensions.append("TF")
        if abs(j_prob - p_prob) < 0.2:
            uncertain_dimensions.append("JP")

        # If no uncertain dimensions found, use dimensions that differentiate top types
        if not uncertain_dimensions:
            for competitor in close_competitors:
                for i, (leader_letter, competitor_letter) in enumerate(zip(current_leader, competitor)):
                    if leader_letter != competitor_letter:
                        dimension_map = {0: "EI", 1: "SN", 2: "TF", 3: "JP"}
                        uncertain_dimensions.append(dimension_map[i])

        # If we found uncertain dimensions, prioritize questions for those dimensions
        best_question_index = remaining_indices[0]  # Default to first question

        if uncertain_dimensions:
            # Find first question that tests the most uncertain dimension
            for index in remaining_indices:
                _, likelihoods = self.questions[index]

                # Get the question type from the first answer's keys
                first_answer_keys = list(likelihoods[1].keys())
                question_dimension = ''.join(first_answer_keys)

                if question_dimension in uncertain_dimensions:
                    best_question_index = index
                    break

        return best_question_index

    def determine_type(self):
        """Find the MBTI type with the highest probability"""
        return max(self.type_probabilities.items(), key=lambda x: x[1])[0]

    def run_test(self):
        print("\n" + "="*50)
        print("Welcome to the Bayesian MBTI Personality Test!")
        print("="*50)
        print("\nThis test uses Bayesian probability to determine your personality type.")
        print("The questions are adaptive, adjusting based on your previous answers.")

        input("\nPress Enter to begin the test...")

        # Track which questions we've asked
        remaining_indices = list(range(len(self.questions)))
        asked_questions = 0

        # Ask up to 10 questions (or fewer if we reach high confidence)
        while remaining_indices and asked_questions < 20:
            # Get the next best question to ask
            question_index: int | None = self.get_next_question_index(remaining_indices)
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

            # If we're very confident (>80%), we could stop early
            if current_probability > 0.8 and asked_questions >= 5:
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

# if __name__ == "__main__":
#     app = BayesianMBTIApp()
#     app.run_test()
