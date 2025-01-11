# AI-Powered Chatbot for Climate Change Policy Data Using RAG and LangChain
An intelligent chatbot is designed to retrieve data and provide customized responses to user inquiries. It is a valuable tool for decision-makers, especially those operating within specific domains, offering timely and relevant information to support their decision-making processes.

## Functionalities
- Backend to integrate with the AI component of the chatbot and the UI using Python Flask.
- Basic UI using HTML, CSS, and JavaScript.
- RAG evaluation.
- NLP model evaluation.
- Chatbot conversation evaluation.
- Fine-tune the summarization model.

## 🔗 Dependencies and Setup

The following tools and libraries are required for setting up the project. 
### Install Python version 3.x

- [x]  Install latest python version from [here.](https://www.python.org) 3.10.12 is recommended 
- [x]  Install pip from [here.](https://pip.pypa.io/en/stable/installation/) 24.3.1 is recommended.
Note: pip will be available as a part of your python installation. you can check the pip version for verifying.
```bash
pip --version
```
### Install the follwoing libs to run Machine Learning Model
- ☑️ torch==2.5.1+cu118
- ☑️ transformers==4.36.0
- ☑️ numpy==1.26.4
- ☑️ flask==3.1.0
- ☑️ elasticsearch==8.17.0
- ☑️ faiss-cpu==1.9.0.post1
- ☑️ spacy==3.8.3
- ☑️ gensim==4.3.3
- ☑️ wikipedia-api==0.7.1
- ☑️ sentencepiece==0.2.0
- ☑️ en-core-web-sm==3.8.0

```bash
pip install -r requirements.txt
```

## 🔗 Run the Chatbot Application and Evaluation
### Run the application inside the intelligent_bot directory.
```bash
cd intelligent_bot
```
```bash
python3 app.py 
```
### Run Jupyter Notebook files in each evaluation directory to see the evaluation results.

## 🔗 Host the Server in CSC
Open http://128.214.253.165:5000/ in browser to load the Chatbot

## 🔗 Microservice Architecture

The Microservice Architecture is used to ensure that the RAG, LangChain, and core framework (backend and UI) of the application are implemented as separate, independent services, adding modularity and scalability to the application.

### Checkout to the branch "adding_microservice"
```bash
git checkout adding_microservice
```

### Run the app.py file in each directory (rag_retrieval_service, langchain_generator_service, core_framework)
```bash
python3 app.py 
```
- rag_retrieval_service - PORT:5002
- langchain_generator_service - PORT:5003
- core_framework - PORT:5000

In here, the core framework calls the langchain_generator_service, which in turn calls the rag_retrieval_service.


