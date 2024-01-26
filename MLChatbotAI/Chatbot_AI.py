import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import pyttsx3
from keras.models import load_model
import os
import json
import random

class ChatBot:
    def __init__(self, model_path=None, words_path=None, classes_path=None, data_path=None):
        if model_path and words_path and classes_path and data_path:
            self.model = self.load_model(model_path)
            self.words, self.classes = self.load_words_classes(words_path, classes_path)
            self.intents = self.load_intents(data_path)
        else:
            # Load default model and data files
            self.model = self.load_model_default()
            self.words, self.classes = self.load_words_classes_default()
            self.intents = self.load_intents_default()

    def load_model_default(self):
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'model', 'chatbot_model2.h5')
        return load_model(model_path)

    def load_model(self, model_path):
        return load_model(model_path)

    def load_intents_default(self):
        current_dir = os.path.dirname(__file__)
        intents_path = os.path.join(current_dir, 'data', 'intents.json')
        with open(intents_path) as file:
            return json.load(file)

    def load_intents(self, data_path):
        with open(data_path) as file:
            return json.load(file)

    def load_words_classes_default(self):
        current_dir = os.path.dirname(__file__)
        words_path = os.path.join(current_dir, 'model', 'words.pkl')
        classes_path = os.path.join(current_dir, 'model', 'classes.pkl')
        with open(words_path, 'rb') as words_file:
            words = pickle.load(words_file)
        with open(classes_path, 'rb') as classes_file:
            classes = pickle.load(classes_file)
        return words, classes

    def load_words_classes(self, words_path, classes_path):
        with open(words_path, 'rb') as words_file:
            words = pickle.load(words_file)
        with open(classes_path, 'rb') as classes_file:
            classes = pickle.load(classes_file)
        return words, classes

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("Found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, sentence):
        ints = self.predict_class(sentence)
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
        return "Sorry, I don't understand that."

    def take_command(self):
        return input("Enter your sentence: ")

    def speak(self, response):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.say(response)
        engine.runAndWait()

#if __name__ == "__main__":
#    model_path = input("Enter the path to your custom model (or press Enter to use default): ").strip()
#    words_path = input("Enter the path to your custom words file (or press Enter to use default): ").strip()
#    classes_path = input("Enter the path to your custom classes file (or press Enter to use default): ").strip()
#    data_path = input("Enter the path to your custom data file (or press Enter to use default): ").strip()
#
#    bot = ChatBot(model_path, words_path, classes_path, data_path)
#
#    while True:
#        user_input = bot.take_command()
#        if user_input.lower() == 'quit':
#            break
#        response = bot.get_response(user_input)
#        print(response)
