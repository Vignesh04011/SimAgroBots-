🌾 SimAgroBots

SimAgroBots is an AI-powered dashboard that predicts crop health and yield using 30 agricultural features. The project leverages machine learning models (Random Forest Classifier and Regressor) to assist farmers and researchers in real-time decision making.

📌 Key Features

* Health classification of crops into Poor, Average, or Good
* Yield prediction in tons per hectare
* Feature importance visualization
* Editable input interface (top 15 primary features + 15 secondary with default average values)
* Encoding explanations for user-friendly interpretation

🚀 Technologies Used

* Python, Streamlit
* scikit-learn for ML modeling
* Pandas, Seaborn, Matplotlib for data analysis and visualization

🤖 ML Models Used

* `health_classifier.pkl`: Random Forest Classifier trained on 30 features to classify crop health
* `yield_regressor.pkl`: Random Forest Regressor trained on 30 features to predict yield (T/Ha)

📁 Model Files

Due to GitHub’s file size limits, model files are hosted on Hugging Face:

* 🔗 [Download health_classifier.pkl](https://huggingface.co/Vignesh0401/SimAgroBots/blob/main/health_classifier.pkl)
* 🔗 [Download yield_regressor.pkl](https://huggingface.co/Vignesh0401/SimAgroBots/blob/main/yield_regressor.pkl)

> ✅ Place both `.pkl` files in the root folder of your project before running the app.

📦 Setup Instructions

1. Clone the repository:
   
   git clone https://github.com/Vignesh04011/SimAgroBots-.git
   cd SimAgroBots-
 

3. (Optional) Create and activate a virtual environment:

   python -m venv venv
   venv\Scripts\activate   # On Windows


4. Install the dependencies:

   pip install -r requirements.txt


5. Download and place the `.pkl` files in the project root:

   * [Download health_classifier.pkl](https://huggingface.co/Vignesh0401/SimAgroBots/blob/main/health_classifier.pkl)
   * [Download yield_regressor.pkl](https://huggingface.co/Vignesh0401/SimAgroBots/blob/main/yield_regressor.pkl)

6. Run the dashboard:

   streamlit run app.py

## 🧠 Encodings Explained

Some features are label encoded for model compatibility. The dashboard provides an explanation section where users can understand the encoded values (e.g., 1 = Loamy, 2 = Sandy, etc.).

