# AI-Powered Chatbot for Climate Change Policy Data Using RAG and LangChain
An intelligent chatbot is designed to retrieve data and provide customized responses to user inquiries. It is a valuable tool for decision-makers, especially those operating within specific domains, offering timely and relevant information to support their decision-making processes.

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
### Run the application inside the intelligent_bot directory
```bash
cd intelligent_bot
```
```bash
python3 app.py 
```
### Run Jupyter Notebook files in each evaluation directory.

## 🔗 Connection to the MQTT broker

## Install libraries

- ☑️ micropython-umqtt.simple==1.3.4
- ☑️ influxdb-client==2.7.11

Adding the properties to the [config.py](pico-code/config.py) should establish connectivity between clients and the broker. In HiveMQ broker, the broker details are under the `OVERVIEW` tab. The `MQTT_USER` and the `MQTT_PWD` are the web client details entered under the `WEB CLIENT` to connect the clients to the server. 

```python
MQTT_BROKER = '' # broker/server URL
MQTT_PORT= 8883
MQTT_USER = ''  # access username
MQTT_PWD = '' # access password
```
Connect on Web Client page using username and password, subscribe all messages.

The connectivity should be established in the MIT app inventor, InfluxDB, and Grafana by changing the properties of the MQTT extension. The same attributes should be used. 

## 🔗 Node-RED and InfluxDB

### Node-red: run command prompt as admin: 
```bash
node-red 
```
Open http://localhost:1880/ in browser

### Influxdb: run command prompt as admin: 
```bash
cd "C:\Program Files\InfluxDB"
```
```bash
influxd
```
Open http://127.0.0.1:8086/ in browser


