import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from sklearn.pipeline import make_pipeline
import joblib
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('fake_job_postings.csv')

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

df['processed_text'] = df['title'].apply(preprocess_text) + ' ' + \
                       df['location'].apply(preprocess_text) + ' ' + \
                       df['department'].apply(preprocess_text) + ' ' + \
                       df['salary_range'].apply(preprocess_text) + ' ' + \
                       df['company_profile'].apply(preprocess_text) + ' ' + \
                       df['description'].apply(preprocess_text) + ' ' + \
                       df['requirements'].apply(preprocess_text) + ' ' + \
                       df['benefits'].apply(preprocess_text)+ ' '+ \
                        df['telecommuting'].apply(preprocess_text)+ ' '+ \
                       df['has_company_logo'].apply(preprocess_text)+ ' '+ \
                       df['has_questions'].apply(preprocess_text)+ ' '+ \
                       df['employment_type'].apply(preprocess_text)+ ' '+ \
                       df['required_experience'].apply(preprocess_text)+ ' '+ \
                       df['required_education'].apply(preprocess_text)+ ' '+ \
                       df['industry'].apply(preprocess_text)+ ' '+ \
                       df['function'].apply(preprocess_text)

X = df['processed_text']
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000),
    RandomForestClassifier(class_weight=class_weights_dict)
)

param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [None, 10, 20, 30],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

joblib.dump(best_model, 'best_job_classifier_model.pkl')

print("Model training complete and saved as 'best_job_classifier_model.pkl'")

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))
