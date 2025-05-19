from flask import Flask, render_template, request, session, redirect, url_for
from app import BayesianMBTIApp
import os
import random
from result_job import MBTIJobPredictor

## CONFIGURABLE ##
question_path = "./data/question-ayspro.csv"
question_job_path = "./data/raw_mbti-indo.csv"
question_count = 40
## CONFIGURABLE ##

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session

# Initialize the MBTI app
mbti_app = BayesianMBTIApp(question_path)

useless_data = {
    "EI": 0,
    "SN": 0,
    "TF": 0,
    "JP": 0,
}

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    # Initialize test if it's the first visit
    if 'remaining_indices' not in session:
        session['remaining_indices'] = list(range(len(mbti_app.questions)))
        session['asked_questions'] = 0
        session['probabilities'] = {mbti_type: 1/16 for mbti_type in mbti_app.mbti_types}
        session['current_question_index'] = None

    if request.method == 'GET':
        try:
            name = request.args['name']
            email = request.args['email']
            telephone = request.args['telp']
            mbti_app.set_email(email)
            mbti_app.set_name(name)
            mbti_app.set_telp(telephone)
        except:
            pass

    # Process answer if POST
    if request.method == 'POST' and 'answer' in request.form:
        answer = int(request.form['answer'])
        question_index = session['current_question_index']

        # Get the question and likelihoods
        if question_index is None:
            return redirect(url_for('index'))
        question, likelihoods, _ = mbti_app.questions[question_index]

        # Update probabilities based on the answer
        probabilities = session['probabilities']
        new_probabilities = {}

        # Apply Bayesian update
        for mbti_type in mbti_app.mbti_types:
            type_likelihood = 1.0
            answer_likelihoods = likelihoods[answer]

            for letter in mbti_type:
                if letter in answer_likelihoods:
                    type_likelihood *= answer_likelihoods[letter]

            new_probabilities[mbti_type] = type_likelihood * probabilities[mbti_type]

        # Normalize probabilities
        total = sum(new_probabilities.values())
        if total > 0:
            for mbti_type in mbti_app.mbti_types:
                new_probabilities[mbti_type] = new_probabilities[mbti_type] / total

        session['probabilities'] = new_probabilities

        # Update remaining questions and count
        remaining_indices = session['remaining_indices']
        # Add safety check before removing
        if question_index in remaining_indices:
            remaining_indices.remove(question_index)
        session['remaining_indices'] = remaining_indices
        session['asked_questions'] = session['asked_questions'] + 1

        # Check if we should end the test
        if len(remaining_indices) == 0 or session['asked_questions'] >= question_count:
            return redirect(url_for('results'))

    # Get next question
    if session['remaining_indices']:
        # Set a limit for attempts to avoid infinite loop
        max_attempts = 10
        attempts = 0
        found_question = False
        global_question = None

        while attempts < max_attempts and not found_question:
            random_index = random.randint(0, len(session['remaining_indices']) - 1)
            session['current_question_index'] = session['remaining_indices'][random_index]
            question, _, y = mbti_app.questions[session['current_question_index']]

            # Check if we've asked enough questions of this type
            if useless_data[y] < question_count // 4:
                useless_data[y] += 1
                global_question = question
                found_question = True
            else:
                attempts += 1

        # If we couldn't find a balanced question after max attempts, just use the current one
        if not found_question:
            session['current_question_index'] = session['remaining_indices'][0]
            question, _, y = mbti_app.questions[session['current_question_index']]
            global_question = question
            useless_data[y] += 1
        print(useless_data)

        return render_template(
                'test.html',
                question=global_question,
                question_num=session['asked_questions'] + 1,
                total_questions=question_count,
                )
    else:
        return redirect(url_for('results'))

@app.route('/results')
def results():
    if 'probabilities' not in session:
        return redirect(url_for('index'))

    # Get the personality type with highest probability
    probabilities = session['probabilities']
    personality_type = max(probabilities.items(), key=lambda x: x[1])[0]
    confidence = probabilities[personality_type] * 100

    # Calculate dimensional breakdown
    e_prob = sum(probabilities[t] for t in mbti_app.mbti_types if t[0] == 'E') * 100
    i_prob = sum(probabilities[t] for t in mbti_app.mbti_types if t[0] == 'I') * 100
    s_prob = sum(probabilities[t] for t in mbti_app.mbti_types if t[1] == 'S') * 100
    n_prob = sum(probabilities[t] for t in mbti_app.mbti_types if t[1] == 'N') * 100
    t_prob = sum(probabilities[t] for t in mbti_app.mbti_types if t[2] == 'T') * 100
    f_prob = sum(probabilities[t] for t in mbti_app.mbti_types if t[2] == 'F') * 100
    j_prob = sum(probabilities[t] for t in mbti_app.mbti_types if t[3] == 'J') * 100
    p_prob = sum(probabilities[t] for t in mbti_app.mbti_types if t[3] == 'P') * 100

    # Get sorted types for display
    sorted_types = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    sorted_types = [(t, round(p*100, 1)) for t, p in sorted_types]

    buffer = ""
    with open(question_job_path, 'r', encoding='utf-8') as f:
        csv_data = f.read()
        predictor = MBTIJobPredictor(csv_data)

        # only take top 3
        sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_3 = dict(sorted_probs[:3])

        predictions = predictor.predict_jobs(top_3, top_n=3)

        for _, (job, score) in enumerate(predictions, 1):
            # buffer += f"{job} ({score:.2f}), "
            buffer += f"{job}, "

        buffer.rstrip()
        buffer = buffer[:-2]

    return render_template(
            'results.html',
            personality_type=personality_type,
            confidence=round(confidence, 1),
            description=buffer,
            description2=mbti_app.personality_descriptions.get(personality_type, ""),
            # description=mbti_app.personality_descriptions.get(personality_type, ""),
            sorted_types=sorted_types[:5],
            dimensions={
                'E': round(e_prob, 1), 'I': round(i_prob, 1),
                'S': round(s_prob, 1), 'N': round(n_prob, 1),
                'T': round(t_prob, 1), 'F': round(f_prob, 1),
                'J': round(j_prob, 1), 'P': round(p_prob, 1)
                },
            user_name=mbti_app.name,
            user_email=mbti_app.email,
            user_telp=mbti_app.telp,
            )

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))
