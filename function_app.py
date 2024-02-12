import azure.functions as func
import logging
import openai
from openai import AzureOpenAI
import requests
import json, os
import traceback
from elasticsearch import Elasticsearch
from langchain.memory import ElasticsearchChatMessageHistory
# from langchain.vectorstores import ElasticsearchStore
from langchain_community.vectorstores import ElasticsearchStore
from es_client import ES_Client, textExpansion_Search, RetrieveESresults
from llm import AzureOpenAi_Client, ChatCompletion, ChatCompletionStream, GeneratedResponse
from text_normalizer import normalize_text



import logging
logging.basicConfig(
    format="%(asctime)s - %(module)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.WARN
)

# Load Env Variables
LLM_TYPE = os.getenv("LLM_TYPE")
ES_INDEX = os.getenv("ES_INDEX")
top_n_results=os.getenv("top_n_results")

logging.info('LLM_TYPE: {0}'.format(LLM_TYPE))
streaming=False

# print in color
'''
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
'''

# question='how to create a new certificate?'

system_prompt="-- You are an AI assistant which answers user's questions in a concise manner.\n-- Your job is to respond to the question strictly by reference to the Source, a passage of text you will be provided.\n-- If the response content contains pointers, then generate the response in pointers as well but in a concise manner.\n -- always answer in natural human way.\n --Always give concise answers.\n-- If you don't know the answer, just say that you don't know, don't try to make up an answer.\n -- Aim to answer queries using existing conversational context.\n -- DO NOT IGNORE URLs in response"

messages = [{ "role": "system", "content": system_prompt }]

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="/chatbot")
def chatbot(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    input = json.loads(req.get_body())
    

    if input:
        try:
            question=input.get('question')
        except Exception as e:
            error_message = {"error":str(e)}
            return func.HttpResponse(
            json.dumps(error_message),
            status_code=200
            )
    else:
            error_message = {"error":str('no input was received')}
            return func.HttpResponse(
            json.dumps(error_message),
            status_code=400
            )

    if not question:
            error_message = {"error":str('no input was received')}
            return func.HttpResponse(
            json.dumps(error_message),
            status_code=400
            )
    
    else:
        if len(question) > 1:
        
            user_prompt = {
            "role": "user",
            "content": question+'. provide the http or https URLs that are received in assistant content. Also if the response contains a list then print each item in new line. Also encourage to raise a service request if the answer is unsatisfactory'
                }

            logging.info('User Question Prompt:\n{0}'.format(user_prompt['content']))

            messages.append(user_prompt)

            # get top results from ES semantic search
            top_hit_responses=RetrieveESresults(question, ES_INDEX, top_n_results)

            final_result_set={}
            if len(top_hit_responses)>0:
                final_result_set['top_source']=top_hit_responses[0]['doc_id']
                final_result_set['top_score']=top_hit_responses[0]['score']
                final_result_set['top_answer']=top_hit_responses[0]['body']
                final_result_set['second_best_source']=top_hit_responses[1]['doc_id']
                final_result_set['second_best_score']=top_hit_responses[1]['score']
                #print(type(final_result_set))
                #print(final_result_set['top_answer'])
                output_answer=normalize_text(final_result_set['top_answer'])
                #print(output_answer)
            else:
                output_answer='-- Greet the user. ask the user if you can assist with anything'
                messages[1]['content']=''
                final_result_set['top_source']=None
                final_result_set['top_score']=None

            # Create Assistant content for OpenAI
            Assistant_Content={
                'role': 'assistant',
                'content': output_answer
                }

            logging.info('ES semantic Search Result:\n{0}'.format(Assistant_Content['content']))

            messages.append(Assistant_Content)

            logging.info(messages)

            OpenAIoutput=GeneratedResponse(messages,is_stream=False)
            logging.info('OpenAI Results::\n\t{0}'.format(str(OpenAIoutput)))

            ai_response={
                            'output': OpenAIoutput['content'], 
                            'source': final_result_set['top_source']
                        }

            return func.HttpResponse(
                 json.dumps(ai_response),
                status_code=200
            )
            
        
        return func.HttpResponse(
                 json.dumps({
                            'output': 'Sorry, but I could not idntify a valid query. Please try again', 
                            'source': None
                        }),
                status_code=200
        )