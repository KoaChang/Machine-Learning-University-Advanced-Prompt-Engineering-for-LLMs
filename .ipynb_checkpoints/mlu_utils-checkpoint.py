import re
import random
import numpy as np
from collections import Counter
import json
import logging
import boto3


from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate

from IPython.display import Markdown, display


def generate_text_mistral(model_id, body):
    """
    Generate text using a Mistral AI model.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        JSON: The response from the model.
    """

    #logger.info("Generating text with Mistral AI model %s", model_id)

    bedrock = boto3.client(service_name='bedrock-runtime')

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id
    )

    #logger.info("Successfully generated text with Mistral AI model %s", model_id)

    return response


def invoke_mistral(prompt:str, model:str='mistral.mistral-7b-instruct-v0:2', temperature:float=0.0, max_tokens:int=1000, stop_sequences:list=[], n:int=1):
    
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 5
    })

    response = generate_text_mistral(model_id=model, body=body)

    response_body = json.loads(response.get('body').read())

    outputs = response_body.get('outputs')

    '''
    for index, output in enumerate(outputs):

        print(f"Output {index + 1}\n----------")
        print(f"Text:\n{output['text']}\n")
        print(f"Stop reason: {output['stop_reason']}\n")
    '''
    
    return outputs[0]['text']

def generate_text_titan(model_id, body):
    """
    Generate text using Amazon Titan Text models on demand.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        response (json): The response from the model.
    """

    #logger.info("Generating text with Amazon Titan Text model %s", model_id)

    bedrock = boto3.client(service_name='bedrock-runtime')

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())

    finish_reason = response_body.get("error")

    if finish_reason is not None:
        raise ImageError(f"Text generation error. Error is {finish_reason}")

    #logger.info("Successfully generated text with Amazon Titan Text model %s", model_id)

    return response_body


def invoke_titan(prompt:str, model:str="amazon.titan-text-lite-v1", temperature:float=0.0, max_tokens:int=1000, stop_sequences:list=[], n:int=1):
    
    body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "stopSequences": stop_sequences,
                "temperature": temperature,
                "topP": 0.9
            }
        })

    response_body = generate_text_titan(model, body)
    '''
    print(f"Input token count: {response_body['inputTextTokenCount']}")


    for result in response_body['results']:
        print(f"Token count: {result['tokenCount']}")
        print(f"Output text: {result['outputText']}")
        print(f"Completion reason: {result['completionReason']}")
    '''

    return response_body['results'][0]['outputText']


def run_inference(prompt:str, model:str, temperature:float, max_tokens:int=1000, stop_sequences:list=[], n:int=1) -> list:
    """
    Function to run inference with models hosted in Bedrock
    """
    model_provider = model.split(".")[0]
    outputs = []
    for i in range(n):
        if model_provider == "mistral":
            outputs.append(invoke_mistral(prompt, model, temperature, max_tokens, stop_sequences, n))
        else:
            outputs.append(invoke_titan(prompt, model, temperature, max_tokens, stop_sequences, n))
        
    return outputs


def generate_outputs(prompt:str, model:str, temperature:float, max_tokens:int=1000, stop_sequences:list=[], n:int=1) -> list:
    """
    Function to wrap calls to inference functions. 
    """
    outputs = run_inference(prompt, model, temperature, max_tokens, stop_sequences, n)
    return outputs
