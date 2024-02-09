from elasticsearch import Elasticsearch
import requests
import json, os, traceback
from dotenv import load_dotenv, find_dotenv
from logger import LOG, handle_error

load_dotenv(find_dotenv())


ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")
ELSER_MODEL = os.getenv("ELSER_MODEL", ".elser_model_2")
ES_INDEX = os.getenv("ES_INDEX")


def ES_Client():
    try:
        
        ElasticSearch_client = Elasticsearch(
            cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_API_KEY
            )

        print('\n#############################################################\n')
        print('Connected to ElasticSearch:\n Instance: {0}\n Cluster: {1}' \
            .format(ElasticSearch_client.info()['name'], ElasticSearch_client.info()['cluster_name']))
        print('\n#############################################################\n')
        return ElasticSearch_client
    except Exception as e:
        LOG.error(traceback.format_exc())


def textExpansion_Search(input_query, ES_INDEX=ES_INDEX, size=3):
    try:
        response = ES_Client().search(
        index=ES_INDEX, 
        size=size,
        query={
            "text_expansion": {
                "full_text_embeddings": {
                    "model_id":".elser_model_2_linux-x86_64",
                    "model_text":input_query
                    }
                }
            }
        )
        
        return response

    except Exception as e:
        LOG.error(traceback.format_exc())



def RetrieveESresults(input_query, ES_INDEX, top_n_results):
    response = textExpansion_Search(input_query, ES_INDEX,top_n_results)
    LOG.info('textExpansion_Search Result:\n{0}'.format(str(textExpansion_Search)))

    search_results_list=[]
    try:

      for hit in response['hits']['hits']:
          search_results_dict={}
          search_results_dict['doc_id']=hit['_id']
          search_results_dict['score']=hit['_score']
          search_results_dict['title']=hit['_source']['title']
          search_results_dict['body']=hit['_source']['body']
          search_results_list.append(search_results_dict)

      return search_results_list
    except:
        return[]

'''
def get_elasticsearch_chat_message_history(index, session_id):
    return ElasticsearchChatMessageHistory(
        es_connection=ES_Client, index=index, session_id=session_id
    )
'''



# retrieve the status of deployment
'''
status = ElasticSearch_client.ml.get_trained_models(
                model_id=ELSER_MODEL, include="definition_status"
            )
print(status['trained_model_configs'][0]['fully_defined'])
'''



# all_indices = ES_Client.indices.get_alias()

# Get Schema/Mapping of the Index:
#raw_data = ES_Client.indices.get_mapping(index='snow-conn_eon_v1')

# Get document from Index:
#raw_data = ES_Client.get(index="snow-conn_eon_v1", id="1")
#print (raw_data)


# Semantic Search using ES search
'''
raw_data = ES_Client.search(index=ES_INDEX, query={ 'bool': { 'must': [ { 'text_expansion': { 'full_text_embeddings': { 'model_id': '.elser_model_2_linux-x86_64',   'model_text': 'where should we have lunch' } } } ], 'filter': [] } })
json_response=json.dumps((raw_data.body), indent=4)
with open("json_response.json", "w") as outfile:
    outfile.write(json_response)
'''
        

# retrieve deployed ELSER Model 
'''
try:
    print('\nGetting Trained Models from ES\n')
    print(ES_Client.ml.get_trained_models(model_id=ELSER_MODEL))
except Exception as e:
    prRed(e)
'''




'''
url     = 'http://f0571c62200b4d249a4c6750ab7f4716.containerhost:9244/esre_eon/_search'
payload = {
  "_source": {
    "excludes": [
      "full_text_embeddings",
      "full_text"
    ]
  },
  "query": {
    "bool": {
      "must": [
        {
          "text_expansion": {
            "full_text_embeddings": {
              "model_id": ".elser_model_2_linux-x86_64",
              "model_text": "where should we have lunch"
            }
          }
        }
      ],
      "filter": []
    }
  }
}
headers = {'Content-Type': 'application/json'}
res = requests.post(url, data=payload, headers=headers)
'''
