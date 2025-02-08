import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset, only keeping the first two columns
df = pd.read_csv("spam.csv", encoding="latin-1", usecols=[0, 1])

# Rename columns for clarity
df.columns = ["label", "message"]

# Convert labels to numbers (ham = 0, spam = 1)
df["label"] = df["label"].map({"ham": 0, "spam": 1})


# Display the first 5 rows
print(df.head())



# Convert text messages into numerical vectors
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df['message'])  # Feature matrix (text -> numbers)
y = df['label']  # Target labels (0 = ham, 1 = spam)

print(f"Feature matrix shape: {X.shape}")

from sklearn.model_selection import train_test_split

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

from sklearn.naive_bayes import MultinomialNB

# Train a Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

print("✅ Model training complete!")

from sklearn.metrics import accuracy_score

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy:.2%}")

def predict_message(message):
    vectorized_message = vectorizer.transform([message])  # Convert text to numbers
    prediction = model.predict(vectorized_message)[0]  # Get prediction (0 = ham, 1 = spam)
    
    return "🚫 SPAM" if probability > 0.8 else "✅ NOT SPAM"

# Test with new messages
test_messages = [
    "Congratulations! You've won a free iPhone! Click here to claim now.",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been compromised. Verify immediately.",
    "Can you send me the meeting notes from today?",
    "WINNER! You've been selected for a free vacation to the Bahamas!"
]

# Print predictions
for msg in test_messages:
    print(f"Message: {msg} -> Prediction: {predict_message(msg)}")

# Allow user input
while True:
    user_message = input("\nEnter a message to check (or type 'exit' to quit): ")
    if user_message.lower() == "exit":
        break
    print(f"Prediction: {predict_message(user_message)}")
