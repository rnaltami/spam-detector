# 📩 Spam Detector

A machine learning-based spam detector using **TF-IDF vectorization** and **Naïve Bayes**, achieving **98% accuracy** in classifying SMS messages as spam or not spam.

---

## ✨ Features
- ✅ **Processes text messages** (Tokenization, Stopword Removal, TF-IDF Vectorization)
- ✅ **Classifies messages as "Spam" or "Not Spam"**
- ✅ **Trained on real-world SMS spam dataset**
- ✅ **Achieves 98% accuracy in spam detection**
- ✅ **Uses Scikit-Learn’s Multinomial Naïve Bayes Model**

---

## 📃 Dataset
The model is trained on an **SMS spam dataset** containing labeled messages as **ham (not spam) or spam**. The dataset was preprocessed to remove stopwords and vectorized using **TF-IDF**.

---

## 🛠️ Installation & Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/rnaltami/spam-detector.git
cd spam-detector
```

### **2. Create and Activate a Virtual Environment**
```bash
python3 -m venv venv  # Create virtual environment
source venv/bin/activate  # Activate on Mac/Linux
venv\Scripts\activate  # Activate on Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Spam Detector**
```bash
python spam_detector.py
```

---

## 💡 Usage
The script can classify SMS messages as **spam or not spam**.

### **Testing the Model with New Messages**
After running the script, you can enter a message for classification:
```bash
Enter a message to check (or type 'exit' to quit): Congratulations! You've won a free iPhone!
Prediction: 🚫 SPAM
```
```bash
Enter a message to check (or type 'exit' to quit): Hey, are we still on for dinner?
Prediction: ✅ NOT SPAM
```

---

## 🌟 Future Improvements
- Experiment with **Logistic Regression** or **Deep Learning Models** for spam detection.
- Deploy as a **web app** for easier user interaction.
- Expand training data with **new spam trends** for higher accuracy.

---

## 📝 License
This project is licensed under the **MIT License**. Feel free to use and modify it.

---

### 🛠️ Contributing
If you'd like to contribute, feel free to **fork the repo**, create a branch, and submit a PR!

---

👉 **Check out the repository:** [github.com/rnaltami/spam-detector](https://github.com/rnaltami/spam-detector)

