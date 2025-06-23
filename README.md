# Phishing Website Classifier

This is a Streamlit web application that uses an Extreme Learning Machine (ELM) model to predict whether a website is phishing or legitimate based on 30 input features. The app is deployed on Streamlit Community Cloud, and the source code is available here.

## Live Demo
Check out the live version here: [Phishing Detection Hub]  https://phishing-app-8cf49xwdt4kzqbtqvnuqsq.streamlit.app/

## Project Overview
- **Dataset**: UCI Phishing Websites Data Set (ID 327).
- **Model**: Custom ELM Classifier, with optional SVM and Naïve Bayes models (pickled files included).
- **Features**: 30 features such as `having_IP_Address`, `URL_Length`, `SSLfinal_State`, etc.
- **Deployment**: Hosted on Streamlit Community Cloud.

## How to Run Locally

### Prerequisites
- Python 3.7 or higher.
- Git (optional, for cloning the repository).

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JawadYounis/phishing-app.git
   cd phishing-app