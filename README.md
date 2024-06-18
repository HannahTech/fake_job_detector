<p align="center">
  <b>Detect fraudulent job postings with machine learning.</b>
</p>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/Web%20App-Visit%20Now-blue" alt="Web App">
  </a>
</p>

---

## 🚀 Web Application

You can access the web application [here](...).

## 📁 Code Overview

The repository contains the following files:

- `app.py`: Flask web application code for serving the web app.
- `train_model.py`: Python script to train the machine learning model using RandomForestClassifier.
- `best_job_classifier_model.pkl`: Pre-trained machine learning model saved after training.
- `index.html`: HTML template for the home page of the web app.
- `result.html`: HTML template for displaying the result of job verification.
- `style.css`: CSS stylesheet for styling the web app.
- `requirements.txt`: List of Python dependencies required to run the web application.
- `README.md`: Overview of the project.

## Data Visualization
![image](https://github.com/HannahTech/fake_job_detector/assets/81828685/ac442dce-ba32-4338-b298-b8cc646beb18)

for more information:
https://public.tableau.com/app/profile/hengameh.khajehpour/viz/Real_Fake_Job/Dashboard1?publish=yes

## 🛠️ Training the Model

To train the model, run the `train_model.py` script. This script reads data from a CSV file (`fake_job_postings.csv`), preprocesses the text data, and trains a RandomForestClassifier model. The trained model is then saved as `best_job_classifier_model.pkl`.

## 🏃‍♀️ Running the Web App

To run the web application, make sure you have installed the required dependencies listed in `requirements.txt`. Then, execute the `app.py` script. The web app will be accessible at `http://localhost:5000`.

## 📄 Collaborators

This project was collaboratively developed by:

- Hengameh Khajehpour
- Anshika Rastogi
- Sige (Cassie) Liu
- Mahlet Mekonnen Gebeyehu
- Swati Prasad Ade
- Geeta Kumari
