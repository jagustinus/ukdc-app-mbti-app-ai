from flask import Flask, render_template, request, session
from app import BayesianMBTIApp
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session

# Initialize the MBTI app
mbti_app = BayesianMBTIApp()

@app.route('/')
def index():
    session.clear()
    mbti_app.reset()
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    if 'question_index' not in session:
        session['remaining_questions'] = [q for q in mbti_app.questions]
        session['question_index'] = mbti_app.get_next_question_index(session['remaining_questions'])
        session['asked_questions'] = 0

    if request.method == 'POST':
        answer = int(request.form['answer'])
        question_index = session['question_index']
        print(answer)
        session['remaining_questions'].remove(question_index)
        session['asked_questions'] += 1

    return render_template(
            'test.html',
            question=mbti_app.questions[session['question_index']][0],
            question_len=len(mbti_app.questions)
            )

@app.route('/result')
def results():
    return render_template('index.html')
