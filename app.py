import streamlit as st
import pickle
import numpy as np

# Define the ELMClassifier class (same as in your Colab notebook)
class ELMClassifier:
    def __init__(self, n_hidden=1000, activation='sigmoid', random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state

    def _activation(self, X):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'relu':
            return np.maximum(0, X)
        else:
            raise ValueError("Unsupported activation function.")

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        y_onehot = np.zeros((n_samples, len(classes)))
        for idx, label in enumerate(classes):
            y_onehot[:, idx] = (y.ravel() == label).astype(float)
        self.W = np.random.randn(n_features, self.n_hidden)
        self.b = np.random.randn(self.n_hidden)
        H = self._activation(np.dot(X, self.W) + self.b)
        self.beta = np.dot(np.linalg.pinv(H), y_onehot)

    def predict(self, X):
        H = self._activation(np.dot(X, self.W) + self.b)
        y_pred = np.dot(H, self.beta)
        return self.classes_[np.argmax(y_pred, axis=1)]

# Load the pickled scaler and model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('elm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Custom CSS and HTML for enhanced interface
st.markdown(
    """
    <style>
    /* Gradient background inspired by the image */
    .stApp {
        background: linear-gradient(to bottom, #8B6F47, #D9C2A9, #B0C4DE);
        color: white;
    }
    /* Header styling */
    .header {
        text-align: center;
        padding: 20px;
        font-family: 'Arial', sans-serif;
        font-size: 48px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
    }
    /* Navigation bar */
    .nav {
        text-align: center;
        padding: 10px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .nav a {
        color: white;
        text-decoration: none;
        margin: 0 15px;
        font-size: 18px;
        transition: color 0.3s;
    }
    .nav a:hover {
        color: #FFD700; /* Gold hover effect */
    }
    /* Input container */
    .input-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #FFD700;
        color: #333;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFA500; /* Orange hover effect */
    }
    /* Success message styling */
    .stSuccess {
        background-color: rgba(0, 128, 0, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# HTML for header and navigation
st.markdown(
    """
    <div class="header">Phishing Detection</div>

    """,
    unsafe_allow_html=True
)

# Streamlit app content
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.title("Phishing Website Classifier")
st.write("Enter the features below to predict if a website is phishing or legitimate.")

# Create input fields for the 30 features
feature_names = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
    'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
    'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index',
    'Links_pointing_to_page', 'Statistical_report'
]

# Store inputs in a dictionary
inputs = {}
st.header("Input Features")
for feature in feature_names:
    inputs[feature] = st.number_input(
        feature.replace('_', ' '),
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=1.0
    )

# Prediction button
if st.button("Predict"):
    # Convert inputs to array in correct order
    input_array = np.array([[inputs[feature] for feature in feature_names]])
    
    # Preprocess inputs
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Display result
    result = "Phishing" if prediction[0] == -1 else "Legitimate"
    st.success(f"Prediction: **{result}**")

st.markdown('</div>', unsafe_allow_html=True)