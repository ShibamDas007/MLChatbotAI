import json
import nltk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer
import os
import pickle
import random

class ChatBotTrainer:
    def __init__(self, mode='default'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, 'models', 'chatbot_model.h5')
        self.words_path = os.path.join(current_dir, 'models', 'words.pkl')
        self.classes_path = os.path.join(current_dir, 'models', 'classes.pkl')
        self.default_intents_path = os.path.join(current_dir, 'data', 'default_intents.json')

        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?']
        self.mode = mode

    def preprocess_data(self, intents):
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        training = []
        output_empty = [0] * len(self.classes)

        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)

        X = np.array([entry[0] for entry in training])
        y = np.array([entry[1] for entry in training])

        return X, y

    def train_default(self, epochs=100, batch_size=5, verbose=1):
        with open(self.default_intents_path, 'r') as file:
            intents_data = json.load(file)

        return self.train(intents_data, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def train_custom(self, custom_intents, epochs=100, batch_size=5, verbose=1):
        return self.train(custom_intents, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def train_from_file(self, file_path, epochs=100, batch_size=5, verbose=1):
        with open(file_path, 'r') as file:
            intents_data = json.load(file)

        return self.train(intents_data, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def train(self, intents_data, epochs=100, batch_size=5, verbose=1):
        X, y = self.preprocess_data(intents_data)

        model = Sequential([
            Dense(512, input_shape=(len(X[0]),), activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(y[0]), activation='softmax')
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # Save the model and related data
        model.save(self.model_path)
        with open(self.words_path, 'wb') as words_file:
            pickle.dump(self.words, words_file)
        with open(self.classes_path, 'wb') as classes_file:
            pickle.dump(self.classes, classes_file)

        print(f"Chatbot training complete! Epochs: {epochs}, Batch Size: {batch_size}, Verbose: {verbose}")

        return model, self.words, self.classes

# Example usage:
if __name__ == "__main__":
    trainer = ChatBotTrainer()

    # Train the chatbot using the specified intents data file
    trained_model, words, classes = trainer.train_default()

    # Example using custom intents data
    custom_intents_data = {
        "intents": [
            {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey"], "responses": ["Hello! How can I help you?"]},
            # Add more custom intents as needed
        ]
    }
    trainer_custom = ChatBotTrainer(mode='custom')
    trained_model_custom, words_custom, classes_custom = trainer_custom.train_custom(custom_intents_data, epoch=10, batch_size=1,verbose=5)

    # Example using intents data from a file provided by the user during runtime
    file_path = r"C:\Users\user\Desktop\New folder\MLChatbotAI\default_intents.json"
    trainer_file = ChatBotTrainer(mode='file')
    trained_model_file, words_file, classes_file = trainer_file.train_from_file(file_path, epochs=10, batch_size=1,verbose=1)
    