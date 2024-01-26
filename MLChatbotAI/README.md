# MLChatbotAI

MLChatbotAI is a Python package that provides tools for building, training, and deploying machine learning-based chatbots.

## Features

- Build and train chatbot models using custom or inbuilt intents data.
- Flexible preprocessing tools for text data.
- Integration support for various applications.

## Installation

You can install MLChatbotAI using pip:

```bash
pip install MLChatbotAI
```
# This code for use the pre exits model 
```bash
from MLChatbotAI import ChatBot

bot = ChatBot()

user_input = bot.take_command()

response = bot.get_response(user_input)

print(response)
```

# This portion use for models which is train on external data
```bash
custom_model_path = "path/to/custom_model.h5"
custom_words_path = "path/to/custom_words.pkl"
custom_classes_path = "path/to/custom_classes.pkl"

custom_model, custom_words, custom_classes = trainer.load_custom_model(custom_model_path, custom_words_path, custom_classes_path)
user_input_custom = "How are you?"
response_custom = trainer.get_response_using_model(custom_model, custom_words, custom_classes, user_input_custom)
print(response_custom)
```
# Train Chatbot with external data

# This for how can train model with pre built data
```bash
from MLChatbotAI import ChatBotTrainer

trainer = ChatBotTrainer()

trained_model_default, words_default, classes_default = trainer.train_default()
```
# This is example of train model on custom data wihtout giving no path
```bash
custom_intents_data = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey"], "responses": ["Hello! How can I help you?"]},
        # Add more custom intents as needed
    ]
}
trained_model_custom, words_custom, classes_custom = trainer.train_custom(custom_intents_data)
```

# This is code for train model with giving path
```bash
file_path = "path/to/custom_intents_file.json"
trained_model_file, words_file, classes_file = trainer.train_from_file(file_path)
```

In future also try to add some more automotion feature.
