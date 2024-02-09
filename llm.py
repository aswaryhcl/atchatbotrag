import openai
from openai import AzureOpenAI
import requests
import json, os, traceback


import logging
logging.basicConfig(
    format="%(asctime)s - %(module)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.WARN
)


OPENAI_VERSION = os.getenv("OPENAI_VERSION")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_OCP_APIM_SUBSCRIPTION_KEY = os.getenv("OPENAI_OCP_APIM_SUBSCRIPTION_KEY")
OPENAI_GPT_MODEL = os.getenv("OPENAI_GPT_MODEL")


def AzureOpenAi_Client():
    try:
        headers = {
                    "Ocp-Apim-Subscription-Key" : OPENAI_OCP_APIM_SUBSCRIPTION_KEY
                }
        
        AzureOpenAIclient = AzureOpenAI(
            azure_endpoint = OPENAI_BASE_URL, 
            api_key = OPENAI_OCP_APIM_SUBSCRIPTION_KEY,  
            api_version = OPENAI_VERSION,
            default_headers = headers
            )
        logging.info('\n#############################################################\n')
        logging.info('Connected to Azure OpenAI:\n Instance: {0}'.format(AzureOpenAIclient))
        logging.info('\n#############################################################\n')

        return AzureOpenAIclient
    
    except Exception as e:
        logging.error(traceback.format_exc())



def ChatCompletion(messages):
    try:
        response = AzureOpenAi_Client().chat.completions.create(
            model=OPENAI_GPT_MODEL, 
            messages=messages,
            temperature=0
        )
        logging.info('\n############## response Started ################ \n')
        logging.info(response)
        logging.info('\n############## response Complete ################ \n')

        return response

    except Exception as e:
        logging.error(e)


def ChatCompletionStream(messages):
    try:
        collected_chunks=[]
        collected_messages = []

        for chunk in AzureOpenAi_Client().chat.completions.create(
                    model=OPENAI_GPT_MODEL, 
                    messages=messages,
                    temperature=0,
                    stream=True
                ):
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk.choices[0].delta.content  # extract the message
            collected_messages.append(chunk_message)  # save the message
            
            #result = "".join(response).strip()
            #result = result.replace("\n", "")
        return collected_messages

    except Exception as e:
        logging.error(traceback.format_exc())



# Azure OpenAI Chat call using Rest API
def ChatCompletionRestAPI(messages):
    """  
    Returns chat response based on user query. 
    Assumes existing list of messages 
    """

    url = f"{OPENAI_BASE_URL}/openai/deployments/{OPENAI_GPT_MODEL}/chat/completions?api-version={OPENAI_VERSION}"

    headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": OPENAI_OCP_APIM_SUBSCRIPTION_KEY
        }

    data = {
            "messages": messages,
            "temperature" : 0,
        }

    response=requests.post(url, headers=headers, data=json.dumps(data)).json()

    return response



# Azure OpenAI Chat call using Python SDK client
def GeneratedResponse(messages, is_stream=False):
    """  
    Returns chat response based on user query. 
    Assumes existing list of messages 
    """
    try:
        if is_stream:
            response = ChatCompletionStream(messages)
            collected_messages = [m for m in response if m is not None]
            full_reply_content = ''.join([m for m in collected_messages])
            return full_reply_content
        else:
            response = ChatCompletion(messages)
            OpenAiResponse={}
            OpenAiResponse['role']=response.choices[0].message.role
            OpenAiResponse['content']=response.choices[0].message.content
            OpenAiResponse['function_call']=response.choices[0].message.function_call
            OpenAiResponse['prompt_tokens'] = response.usage.prompt_tokens
            OpenAiResponse['output_tokens'] = response.usage.completion_tokens
            return OpenAiResponse
    except Exception as e:
        logging.error(traceback.format_exc())

# Langchain retriever
'''
store = ElasticsearchStore(
    es_connection=ElasticSearch_client,
    index_name=ES_INDEX,
    strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(model_id=ELSER_MODEL),
)

retriever = store.as_retriever()
print(retriever)
question = 'How to request a new certificate?'
#retriever.get_relevant_documents(question)
#docs = store.as_retriever().invoke(question)
#print(docs)
'''