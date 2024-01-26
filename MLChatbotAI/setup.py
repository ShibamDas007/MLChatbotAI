from setuptools import setup, find_packages

setup(
    name='MLChatbotAI',
    version='1.0.0',
    packages=find_packages(),
    description='''
        MLChatbotAI is a Python package that provides tools for building, 
        training, and deploying machine learning-based chatbots. 
        It includes a flexible training module that allows users to preprocess 
        and train chatbot models using custom or inbuilt intents data. 
        The package offers easy-to-use functionality for developers to create 
        and integrate chatbot capabilities into their applications.
    ''',
    long_description_content_type='text/markdown',
    include_package_data=True,
    author='Shibam Das',
    author_email='shibomdas121@gmail.com',
    license='MIT',
    url='https://github.com/ShibamDas007/python-DeepChat_AI',
    package_data={
        'MLChatbotAI': [
            'model/chatbot_model.h5',
            'model/words.pkl',
            'model/classes.pkl',
            'data/default_intents.json',
            'train_chatbot.py',
            'Chatbot_AI.py'
        ]
    },
    install_requires=[
        'pyttsx3>=2.90',
        'nltk>=3.8.1',
    ]
)