import redis
import pickle
import torch.multiprocessing as mp

import torch
from vllm import LLM, SamplingParams
import time
import ray


redis_communication = redis.StrictRedis(host='localhost', port=6379, db=0)
redis_communication.delete('Tasks')
redis_communication.delete('Response')  

ray.init(num_gpus=8 )

# device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5' , 'cuda:6', 'cuda:7'] 
device_list = ['cuda:4', 'cuda:5' , 'cuda:6', 'cuda:7'] 
dir_list = [
    "/home/jiangpeiwen2/jiangpeiwen2/workspace/LLMs/ChatGLM3-6B/ZhipuAI/chatglm3-6b",
    "/home/jiangpeiwen2/jiangpeiwen2/workspace/LLMs/TKGT_Model/e2e_3_50/ZhipuAI/chatglm3-6b"
]

model_dir = dir_list[0]
# 全局变量来保持模型实例
model_instance = None

@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self):
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        global model_instance
        if model_instance is None:
            print("Loading model...")
            self.model_instance = LLM(
                model=model_dir,  # 替换为实际的模型路径
                trust_remote_code=True,
                dtype='float16',
                gpu_memory_utilization=0.7,
                seed=32
            )
            print("Model loaded.")

    def generate(self, prompts):
        index, inputs_list = prompts
        outputs = self.model_instance.generate(inputs_list, self.sampling_params, use_tqdm=False)
        ret = []
        for output in outputs:
            ret.append(output.outputs[0].text)
        return (index, ret)
    

def Inferece( ):
    num_workers = 8
    index_dict = {}
    workers = [ModelWorker.remote() for _ in range(num_workers)]
    print("Listening for requests...")

    while True:
        message = redis_communication.lpop('Tasks')
        if message!=None:
            index, task = pickle.loads(message)
            worker = workers[index % num_workers]
            index_dict[index] = worker.generate.remote((index, task))
    
        for index, _output in index_dict.items():
            if index_dict[index]!=None:
                completed, _ = ray.wait([_output], timeout=0)
                if completed:
                    result = ray.get(_output)
                    redis_communication.rpush('Response', pickle.dumps( result ))
                    index_dict[index]=None

    
if __name__=="__main__":
    Inferece( )