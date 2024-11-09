from transformers import pipeline
from xarray.tutorial import base_url

from utils.llm_utils import *
from utils.prompt import *
import numpy as np
import json
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--task_name',type=str,default='gameof24',choices=['gameof24','checkmate','wordsorting','gs','multi_step','sonnet','mgsm','p3_test'])
parser.add_argument('--api',default=None,type=str,help='input baseApi here')
parser.add_argument('--api_key',default=None,type=str,help='input your api key here')
parser.add_argument('--model_id',type=str,default='gpt-4',help='Input model id here, if use local model, input the path to the local model')
parser.add_argument('--top_one',default=10,type=int,help='Input the number of reversal candidate sets here')

if __name__ == '__main__':
    args = parser.parse_args()
    task = args.task_name
    api_key = args.api_key
    model_id = args.model_id
    top_one = args.top_one

    pipeline=Pipeline(model_id=model_id, base_url=base_url, api_key=api_key, prob=True)
    # user_prompt=benchmark_input[task]
    # demos=user_prompt.split("###Example###")[1].split("###Input###")[0]
    # user_instruction=user_prompt.split("###Example###")[0] #for cpm
    demos="Input:1, 5, 5, 5; Output:5× (5 − 1 ÷ 5) = 24"

    ##Preference-Guided Reverse Reasoning
    llm_taste=rot_pipeline(pipeline,reversal_of_thought,demos=demos,warmup=top_one)
    print(llm_taste)





