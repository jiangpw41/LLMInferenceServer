"""
"""

from openai import OpenAI
import pickle
import os
from tqdm import tqdm
import concurrent
from concurrent.futures import ProcessPoolExecutor

def multi_process( processor, prompt_list, client, model):
    # 所有使用多进程工具的函数，必须将迭代对象及其index作为函数参数的前两位
    splited_task_list = [None]*len(prompt_list)
    with ProcessPoolExecutor() as executor:
        future_to_item = {}
        # 建立进程-任务映射
        for j, item in enumerate(prompt_list):
            future_to_item[executor.submit(processor, item, client, model)] = j 
        # 收获进程结果
        for future in concurrent.futures.as_completed(future_to_item): 
            splited_task_list[ future_to_item[future]] = future.result()
    return splited_task_list

def single_chat( prompt, client, model):
    completion = client.chat.completions.create(
        model= model,  # "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.content

def Inference_offline_api( model, file_input_path, file_output_path, config_data, sample_little=None ):
    print( f"Load prompt list from {file_input_path}")
    with open( file_input_path, 'rb') as f: 
        prompt = pickle.load(f)
    if not os.path.exists(file_output_path):
        os.mkdir(file_output_path)
    if sample_little!=None and isinstance( sample_little, int):
        print(f"进行小样本推理：{sample_little}")
        prompt = prompt[:sample_little]
    client = OpenAI(
        api_key = config_data['API']['api_key'],
        base_url = config_data['API']['base_url'],
    )
    if model not in config_data['API']['model_list']:
        raise Exception( f"模型不支持{model}")
    ret = []
    for i in tqdm( range( len(prompt)), desc=f"API {model} processing"):
        prompt_list = prompt[i]
        predict_list = multi_process( single_chat, prompt_list, client, model)
        ret.append( predict_list )
    with open( os.path.join(file_output_path, "predict_list.pickle" ), 'wb') as f:  
        pickle.dump(ret, f)


        



print(completion.choices[0].message)

def Inference_offline_local( file_input_path, file_output_path, sample_little=None):
    # 加载Prompt，检查本地路径