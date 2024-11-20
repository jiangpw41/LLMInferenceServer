import redis
import pickle
import torch.multiprocessing as mp
from openai import OpenAI
import yaml

def single_chat( prompt, client, model):
    completion = client.chat.completions.create(
        model= model,  # "",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.content

def Inference_online_local( model, config_data ):
    redis_communication = redis.StrictRedis(host='localhost', port=6379, db=0)
    redis_communication.delete('Tasks')
    redis_communication.delete('Response')
    client = OpenAI(
        api_key = config_data['API']['api_key'],
        base_url = config_data['API']['base_url'],
    )
    if model not in config_data['API']['model_list']:
        raise Exception( f"模型不支持{model}")
    print(f"LLM Inderence start! (api)")
    while True:
        message = redis_communication.lpop('Tasks')
        if message!=None:
            index, task = pickle.loads(message)
            if task=="<STOP>":
                print("推理进程根据指令退出")
                redis_communication.rpush('Response', pickle.dumps( (index, "进程已安全退出！") ))
                break
            outputs = single_chat( task, client, model)
            redis_communication.rpush('Response', pickle.dumps( (index, outputs) ))