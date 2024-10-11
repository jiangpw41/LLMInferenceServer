import redis
import pickle
import argparse
import os
from vllm import LLM, SamplingParams

def Inference( model_path, gpu_id ):
    redis_communication = redis.StrictRedis(host='localhost', port=6379, db=0)
    redis_communication.delete('Tasks')
    redis_communication.delete('Response')
    # sampling_params = SamplingParams(temperature=0, top_p=1, use_beam_search=True, best_of=2)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype='float16',
        gpu_memory_utilization=0.7,
        seed = 32,
    )
    print(f"LLM Inderence start at GPU {gpu_id} and you can connect it using {redis.StrictRedis(host='localhost', port=6379, db=0)}!")
    while True:
        message = redis_communication.lpop('Tasks')
        if message!=None:
            index, task = pickle.loads(message)
            outputs = llm.generate(task, sampling_params, use_tqdm=False)   # 将输入提示添加到vLLM引擎的等待队列中，并执行vLLM发动机以生成高吞吐量的输出。输出以RequestOutput对象列表的形式返回，其中包括所有输出令牌。
            ret = []
            for output in outputs:
                ret.append( output.outputs[0].text )
            redis_communication.rpush('Response', pickle.dumps( (index, ret) ))

if __name__=="__main__":
    """
    parser = argparse.ArgumentParser(description="Run script with external arguments")
    parser.add_argument('--model_path', type=str, required=True, help='使用的模型路径')
    parser.add_argument('--gpu_id', type=int, required=True, help='使用的GPU')
    args = parser.parse_args()
    # 获取参数值
    
    model_path = args.model_path
    gpu_id = args.gpu_id
    """
    model_path = "/home/jiangpeiwen2/.cache/modelscope/hub/ZhipuAI/chatglm3-6b"
    gpu_id = 1
    Inference( model_path, gpu_id )