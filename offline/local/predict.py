"""
数据接口定义：
输入prompt_list: List[List]，外部的List长度决定循环次数以及多卡各自的循环次数，内部的List长度是batch_size。
输出predict_list: List[List]，n张卡就有n个部分

函数接口定义：6个参数
part_id             负责的部分的id
part_num            总的部分数量
gpu_id              使用的GPU id
model_path          本地模型路径
file_output_path：  Prompt输入数据路径
file_output_path：  Predict输出结果路径
"""

import pickle
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import sys
import time

def get_interval_pair( prompt, part_id, part_num ):
    total_len = len(prompt)
    part_len = total_len // part_num
    index_list = []
    for i in range(part_num):
        index_list.append( part_len*i )
    index_list.append( total_len)
    # 获取所有分部的range左右区间，并根据part_id获取自身要处理的数据集的左右区间
    split_list = []
    for i in range(part_num):
        index_pair = []
        left = index_list[i]
        right = index_list[i+1]
        split_list.append((left, right))
    index_pair = split_list[part_id]
    return index_pair

def Inference( part_id, part_num, gpu_id, model_path, file_input_path, file_output_path):
    # 加载Prompt，检查本地路径
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print( f"Load prompt list from {file_input_path}")
    with open( file_input_path, 'rb') as f: 
        prompt = pickle.load(f)
    if not os.path.exists(file_output_path):
        os.mkdir(file_output_path)
    # 按照可用GPU进行分治
    index_pair = get_interval_pair( prompt, part_id, part_num)
    # 加载模型，在对应GPU上启动vllm引擎（在bash脚本中由环境变量指定GPU）
    sampling_params = SamplingParams(temperature=0, top_p=1, use_beam_search=True, best_of=2)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype='float16',
        gpu_memory_utilization=0.7,
        seed = 32,
    )
    print(f"No.{part_id} LLM Inderence start!")
    # 开始批处理
    part_result = []
    for i in tqdm(range(index_pair[0], index_pair[1]), dynamic_ncols=True, file=sys.stdout, desc=f"Part {part_id} processing"):
        
        outputs = llm.generate(prompt[i], sampling_params, use_tqdm=False)   # 将输入提示添加到vLLM引擎的等待队列中，并执行vLLM发动机以生成高吞吐量的输出。输出以RequestOutput对象列表的形式返回，其中包括所有输出令牌。
        ret = []
        for output in outputs:
            ret.append( output.outputs[0].text )
        part_result.append( ret )
        """
        part_result.append(i)
        time.sleep(0.1)
        """
    # 使用pickle.dump()保存列表 
    with open( os.path.join(file_output_path, str(part_id) ), 'wb') as f:  
        pickle.dump(part_result, f)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Run script with external arguments")
    parser.add_argument('--part_id', type=int, required=True, help='本进程负责的部分和GPU编号')
    parser.add_argument('--part_num', type=int, required=True, help='总的GPU数量')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU索引')
    parser.add_argument('--model_path', type=str, required=True, help='使用的模型路径')
    parser.add_argument('--file_input_path', type=str, required=True, help='提示词列表所在路径')
    parser.add_argument('--file_output_path', type=str, required=True, help='预测结果输出路径')

    args = parser.parse_args()
    # 获取参数值
    part_id = args.part_id
    part_num = args.part_num
    model_path = args.model_path
    file_input_path = args.file_input_path
    file_output_path = args.file_output_path
    gpu_id = args.gpu_id
    '''
    part_id = 0
    part_num = 1
    model_path =
    file_input_path = 
    file_output_path = 
    '''
    Inference( part_id, part_num, gpu_id, model_path, file_input_path, file_output_path )