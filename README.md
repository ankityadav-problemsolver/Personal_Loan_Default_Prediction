# 💰 Personal Loan Default Prediction - Streamlit Application

> **Live Demo**: [Click here to try the app 🚀](https://personalloanprediction.streamlit.app/)

---

![Animated Header](assets/screenshots/header_animation.gif)

---

## 🧹 Problem Statement

Financial institutions face increasing challenges in accurately identifying applicants who might default on personal loans. Misclassification can lead to bad debts, regulatory issues, and loss of trust. A reliable prediction system can help mitigate risk, enhance operational efficiency, and drive smarter lending decisions.

---

## 🎯 Goal of the Project

- ✅ Predict the probability of loan default based on customer attributes
- ✅ Provide a production-ready, interactive Streamlit app for banks and analysts
- ✅ Explain predictions with SHAP visualizations
- ✅ Export reports to share insights with decision-makers

---

## 🛠️ Flow Diagram

```mermaid
graph TD
A[User Input Form] --> B[Data Preprocessing]
B --> C[Prediction Engine (ML Model)]
C --> D[Risk Score + Approval Decision]
D --> E[Explainability + Export Report]
```

---

## 📂 Folder Structure

```
📆 personal-loan-prediction
├── app.py                    # Main Streamlit app
├── model/
│   └── model.pkl             # Trained ML model
├── data/
│   └── sample_data.csv       # Sample training data
├── utils/
│   └── preprocessing.py      # Preprocessing functions
├── assets/
│   └── screenshots/          # UI screenshots & animations
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
```

---

## 🛠️ Tech Stack

| Layer             | Technology                      |
|------------------|----------------------------------|
| **Frontend**     | Streamlit                        |
| **Backend**      | Python                           |
| **ML Models**    | Logistic Regression, Random Forest, XGBoost |
| **Visualization**| Plotly, Matplotlib, SHAP         |
| **Export Tools** | PDFKit, Streamlit Export         |

---

## 🧠 Code Walkthrough

```python
# Load the model
model = pickle.load(open("model/model.pkl", "rb"))

# Preprocess user inputs
def preprocess(data):
    ...
    return processed_input

# Make prediction
prediction = model.predict(processed_input)

# Show results
st.success(f"Prediction: {'Approved' if prediction==0 else 'Rejected'}")
```

---

## 🞼 Application UI & Screenshots

### 🔘 Home Page
![Home Page](assets/screenshots/home.png)

### 🔢 Prediction Result
![Prediction Result](assets/screenshots/result.png)

### 📉 SHAP Explainability
![Explainability](assets/screenshots/shap_output.png)

### 🎟️ Animated Workflow
![Animated Demo](assets/screenshots/animated_demo.gif)

---

## ✨ Key Features

- 🔹 User-friendly UI built in Streamlit
- 🔹 Real-time loan eligibility prediction
- 🔹 Support for multiple ML models
- 🔹 SHAP visualizations for explainability
- 🔹 Exportable PDF reports
- 🔹 Clean, minimal, responsive design

---

## 🔮 Scientific Innovation

> What makes this project stand out:

- 🧠 **Explainable AI (XAI)**: Integrates SHAP values to explain each prediction
- 🧬 **Bias Check Module (Coming Soon)**: Identify model fairness issues across genders/ages
- ✨ **Risk Interpretation Layer**: Converts numeric predictions into easy-to-understand advice
- 🚀 **Model Comparator**: Visual comparison between Logistic, Random Forest & XGBoost (Planned)

---

## 🚀 Future Enhancements

- 🚀 Aadhaar/PAN Verification via Gemini API
- 🎮 Voice-based AI Assistant for customer service
- 🤖 Chatbot integration for insights
- 📊 Historical dashboard for tracking applicant trends
- 👨‍📈 Credit score simulator based on customer profile edits

---

## 🚧 How to Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/personal-loan-prediction.git
cd personal-loan-prediction

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the application
streamlit run app.py
```

---

## 🧳 Collaboration & Contribution

We'd love your help! You can:

- ✨ Improve UI/UX
- 🧠 Optimize the model pipeline
- 📉 Add new data sources or APIs
- 🤝 Translate into local languages

### How to Contribute
```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/new-awesome-feature

# Commit and push your changes
git commit -m "Added awesome feature"
git push origin feature/new-awesome-feature

# Submit a Pull Request
```

---

## 📢 Contact Me

| Platform       | Link                                    |
|----------------|------------------------------------------|
| 💼 LinkedIn    | [Ankit Sharma](https://www.linkedin.com/in/yourlinkedin/) |
| 💻 GitHub      | [ankitsharma](https://github.com/yourusername)             |
| 📧 Email       | ankit.yourmail@example.com                 |

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to fork, adapt, and build upon it for your use case!

---

## 💎 Pro Tip

Use this architecture as a blueprint for any classification-based real-time prediction system with SHAP explainability, PDF reporting, and clean UI design.

---

## 🔹 Tags
`#LoanPrediction` `#StreamlitApp` `#ExplainableAI` `#SHAP` `#FinanceAI` `#BankingML` `#CreditScoring`

---
