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
from MLChatbotAI import ChatBot

# Create an instance of the ChatBot
bot = ChatBot()

# Get user input
user_input = bot.take_command()

# Get the bot's response
response = bot.get_response(user_input)

# Print or use the response as needed
print(response)

# Using a Custom Model
custom_model_path = "path/to/custom_model.h5"
custom_words_path = "path/to/custom_words.pkl"
custom_classes_path = "path/to/custom_classes.pkl"

custom_model, custom_words, custom_classes = trainer.load_custom_model(custom_model_path, custom_words_path, custom_classes_path)
user_input_custom = "How are you?"
response_custom = trainer.get_response_using_model(custom_model, custom_words, custom_classes, user_input_custom)
print(response_custom)
```



```
from MLChatbotAI import ChatBotTrainer

# Create an instance of the ChatBotTrainer
trainer = ChatBotTrainer()

# Train the default model using built-in intents data
trained_model_default, words_default, classes_default = trainer.train_default()

# Example using custom intents data
custom_intents_data = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey"], "responses": ["Hello! How can I help you?"]},
        # Add more custom intents as needed
    ]
}
trained_model_custom, words_custom, classes_custom = trainer.train_custom(custom_intents_data)

# Example using intents data from a file provided by the user during runtime
file_path = "path/to/custom_intents_file.json"
trained_model_file, words_file, classes_file = trainer.train_from_file(file_path)
```