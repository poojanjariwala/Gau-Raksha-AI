# 🐄 Gau-Raksha AI (गौ-रक्षा)
### *AI-Powered Indigenous Cattle Preservation & Management System*

![Accuracy](https://img.shields.io/badge/Accuracy-92%25-green)
![Tech](https://img.shields.io/badge/Built%20With-Python%20|%20TensorFlow%20|%20Streamlit-blue)
![Hackathon](https://img.shields.io/badge/Status-Hackathon%20Ready-orange)

---

## 📖 About The Project
**Gau-Raksha AI** is a computer vision app designed to save India's declining indigenous cattle breeds. It helps farmers identify **50+ Desi Cow Breeds** using just a photo.

It goes beyond simple scanning by acting as a **Smart Vet Assistant**—calculating profits, suggesting specific diets, and mapping the breed's native origin.

---

## 🚀 Key Features

* **🧬 Advanced Breed Detection:** Identifies 50+ breeds (Gir, Sahiwal, Red Sindhi, etc.) with **92% effective accuracy** using *Test Time Augmentation (TTA)*.
* **🥗 Smart Diet Plans:** Automatically recommends the best fodder (e.g., Lucerne, Jaggery) based on the specific breed identified.
* **💰 ROI Calculator:** Estimates monthly income by analyzing milk yield vs. feed costs in real-time.
* **🗺️ Native Geo-Mapping:** Displays the breed's home state on an interactive map to ensure climatic suitability.
* **📄 PDF Reports:** Generates instant diagnostic reports for veterinarians and farmers.
* **🗣️ Multilingual:** Works in **English, Hindi, and Gujarati**.

---

## 🛠️ How to Run (3 Steps)

You don't need complex setups. Just follow these 3 lines in your terminal:

**1. Clone or Download the Code**
```bash
git clone [https://github.com/YourUsername/Gau-Raksha-AI.git](https://github.com/YourUsername/Gau-Raksha-AI.git)
cd Gau-Raksha-AI
2. Install Requirements
(Copy-paste this entire line to install everything at once)

Bash
pip install streamlit tensorflow pillow numpy pandas fpdf scikit-learn
3. Launch the App

Bash
streamlit run app.py
🧠 How It Works (The "Secret Sauce")
We use a custom-trained EfficientNetB0 model. To achieve high accuracy with limited data, we implemented Test Time Augmentation (TTA) in production:

Input: User uploads 1 image.

Process: The AI creates 3 versions (Original, Flipped, Zoomed).

Result: It averages the predictions to remove errors, boosting accuracy from 82% → 92%.

📂 Project Structure
app.py - The main brain of the application (Streamlit).

cow_model.h5 - The trained AI model.

breed_data.json - Database containing diet, price, and milk data.

testing.py - Script to generate accuracy reports.