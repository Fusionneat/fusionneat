import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load dataset
with open("dataset.json", "r") as file:
    data = json.load(file)

# Extract queries and intents
queries = [item["query"] for item in data["data"]]
intents = [item["intent"] for item in data["data"]]
responses = {item["intent"]: item["response"] for item in data["data"]}

# Encode intents
label_encoder = LabelEncoder()
encoded_intents = label_encoder.fit_transform(intents)

# Tokenize and pad queries
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(queries)
sequences = tokenizer.texts_to_sequences(queries)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, encoded_intents, test_size=0.2, random_state=42
)

# Convert labels to categorical
num_classes = len(set(encoded_intents))
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Build the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save("chatbot_model.h5")
print("Model saved as chatbot_model.h5")

# Save tokenizer and label encoder
with open("tokenizer.pkl", "wb") as t_file:
    pickle.dump(tokenizer, t_file)

with open("label_encoder.pkl", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

print("Tokenizer and Label Encoder saved successfully.")

# Function to predict and respond
def get_response(query):
    sequence = tokenizer.texts_to_sequences([query])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post")
    prediction = model.predict(padded)
    predicted_intent = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return responses.get(predicted_intent, "Sorry, I couldn't understand your question.")

# Test the system
example_query = "Can you help with website migration?"
print(f"Query: {example_query}")
print(f"Response: {get_response(example_query)}")
