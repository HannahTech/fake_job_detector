from flask import Flask, render_template, request
import joblib
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

model = joblib.load('best_job_classifier_model.pkl')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        processed_text = ' '.join(tokens)
        return processed_text
    else:
        return ''

def shorten_text(text):
    if text is None:
        return ''
    words = text.split()
    shortened_text = ' '.join(words[:5])
    if len(words) > 5:
        shortened_text += ' ...'
    return shortened_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    job_title = request.form.get('job_title')
    location = request.form.get('location')
    department = request.form.get('department')
    salary_range = request.form.get('salary_range')
    company_profile = request.form.get('company_profile')
    description = request.form.get('description')
    requirements = request.form.get('requirements')
    benefits = request.form.get('benefits')
    telecommuting = request.form.get('telecommuting')
    has_logo = request.form.get('has_logo')
    has_questions = request.form.get('has_questions')
    employment_type = request.form.get('employment_type')
    required_experience = request.form.get('required_experience')
    required_education = request.form.get('required_education')
    industry = request.form.get('industry')
    function = request.form.get('function')

    input_text = preprocess_text(f"{job_title} {location} {department} {salary_range} {company_profile} {description} {requirements} {benefits} {telecommuting} {has_logo} {has_questions} {employment_type} {required_experience} {required_education} {industry} {function}")

    prediction = model.predict([input_text])[0]

    result = "Real" if prediction == 0 else "Fake"

    return render_template('result.html', 
                           job_title=shorten_text(job_title),
                           location=shorten_text(location),
                           department=shorten_text(department),
                           salary_range=shorten_text(salary_range),
                           company_profile=shorten_text(company_profile),
                           description=shorten_text(description),
                           requirements=shorten_text(requirements),
                           benefits=shorten_text(benefits),
                           telecommuting=shorten_text(telecommuting),
                           has_logo="Yes" if has_logo == '1' else "No" if has_logo == '0' else None,
                           has_questions="Yes" if has_questions == '1' else "No" if has_questions == '0' else None,
                           employment_type=shorten_text(employment_type),
                           required_experience=shorten_text(required_experience),
                           required_education=shorten_text(required_education),
                           industry=shorten_text(industry),
                           function=shorten_text(function),                           
                           result=result)

if __name__ == '__main__':
    app.run(debug=True)
