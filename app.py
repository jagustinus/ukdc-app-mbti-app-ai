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
            "ISTJ": "Methodical and dependable, ISTJs thrive in roles like software engineer, systems analyst, or IT administrator, where precision, structure, and long-term reliability are valued.",
            "ISFJ": "Responsible and thorough, ISFJs excel in QA testing, technical support, or documentation, where careful follow-through and user-centric thinking are key.",
            "INFJ": "Purpose-driven and insightful, INFJs may enjoy UX design, AI ethics, or data science for social good, where abstract thinking meets meaningful impact.",
            "INTJ": "Strategic and independent, INTJs are well-suited for software architecture, machine learning, or startup tech leadership, where vision and self-motivation are critical.",
            "ISTP": "Analytical and hands-on, ISTPs thrive as embedded systems engineers, cybersecurity specialists, or DevOps engineers, where real-time problem-solving is required.",
            "ISFP": "Sensitive and observant, ISFPs may enjoy UI/UX design, front-end development, or digital media, where aesthetics and user experience are vital.",
            "INFP": "Idealistic and imaginative, INFPs may be drawn to creative coding, game development, or human-centered data science, where values and vision align with technology.",
            "INTP": "Curious and conceptual, INTPs fit well in AI research, algorithm development, or open-source projects, where independent exploration and innovation are encouraged.",
            "ESTP": "Action-oriented and adaptable, ESTPs might thrive in tech sales, startup operations, or field engineering, where quick thinking and flexibility are key.",
            "ESFP": "Energetic and sociable, ESFPs may enjoy roles in tech support, digital marketing, or community engagement for tech platforms, where interaction and enthusiasm are assets.",
            "ENFP": "Creative and enthusiastic, ENFPs flourish in startup environments, product design, or innovation labs, where vision, communication, and adaptability matter.",
            "ENTP": "Inventive and dynamic, ENTPs are often great in entrepreneurship, product management, or emerging tech research, where bold ideas and quick learning are beneficial.",
            "ESTJ": "Organized and decisive, ESTJs are well-suited for project management, technical leadership, or IT operations, where systems, structure, and execution are essential.",
            "ESFJ": "Cooperative and supportive, ESFJs may thrive in user training, customer success roles, or HR tech, where empathy and order intersect.",
            "ENFJ": "Empathetic and motivating, ENFJs do well in team leadership, edtech, or user research, where communication and advocacy help guide technology use.",
            "ENTJ": "Visionary and assertive, ENTJs are ideal for CTO roles, tech strategy, or product ownership, where leadership and long-term planning are vital."
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

        return remaining_indices[0]

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
            # if current_probability > 0.8 and asked_questions >= 5:
            #     print("\nHigh confidence reached, finishing test early...")
            #     break

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

    def reset(self):
        for mbti_type in self.mbti_types:
            self.type_probabilities[mbti_type] = 1/16
