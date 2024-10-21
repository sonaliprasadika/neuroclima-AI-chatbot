from elasticsearch import Elasticsearch

es = Elasticsearch("http://195.148.31.180:9200")
# Test the connection
try:
    info = es.info().body
    print(info)
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")