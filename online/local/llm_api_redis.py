import redis
import pickle
import torch.multiprocessing as mp
import argparse
import os
from vllm import LLM, SamplingParams


def Inference_online_local( model_dir, gpu_id):
    redis_communication = redis.StrictRedis(host='localhost', port=6379, db=0)
    redis_communication.delete('Tasks')
    redis_communication.delete('Response')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # sampling_params = SamplingParams(temperature=0, top_p=1, use_beam_search=True, best_of=2)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        dtype='float16',
        gpu_memory_utilization=0.7,
        seed = 32,
    )   # 必须使用模型支持列表中的模型名
    print(f"LLM Inderence start!")
    while True:
        message = redis_communication.lpop('Tasks')
        if message!=None:
            index, task = pickle.loads(message)
            if task=="<STOP>":
                print("推理进程根据指令退出")
                redis_communication.rpush('Response', pickle.dumps( (index, "进程已安全退出！") ))
                break
            outputs = llm.generate(task, sampling_params, use_tqdm=False)   # 将输入提示添加到vLLM引擎的等待队列中，并执行vLLM发动机以生成高吞吐量的输出。输出以RequestOutput对象列表的形式返回，其中包括所有输出令牌。
            ret = []
            for output in outputs:
                ret.append( output.outputs[0].text )
            redis_communication.rpush('Response', pickle.dumps( (index, ret) ))

if __name__=="__main__":
    Inference( "" )