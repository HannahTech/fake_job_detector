from flask import Flask, render_template, request

app = Flask(__name__)

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
    has_logo = request.form.get('has_logo')
    has_questions = request.form.get('has_questions')

    result = "Fake"

    return render_template('result.html', job_title=job_title,
                           location=location,
                           department=department,
                           salary_range=salary_range,
                           company_profile=company_profile,
                           description=description,
                           requirements=requirements,
                           benefits=benefits,
                           has_logo=has_logo,
                           has_questions=has_questions,result=result)

if __name__ == '__main__':
    app.run(debug=True)
