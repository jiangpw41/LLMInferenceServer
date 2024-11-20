import argparse
import os
import sys  
import yaml
import sys
import shutil
import multiprocessing

_ROOT_PATH = os.path.dirname( os.path.abspath(__file__) )
sys.path.insert( 0, _ROOT_PATH)
from utils import merge_predict, get_gpu_list
from offline.local.predict import Inference_offline_local
from offline.api.predict import Inference_offline_api
from online.local.llm_api_redis import Inference_online_local
from online.api.llm_api_redis import Inference_online_api

with open( os.path.join( _ROOT_PATH, 'config.yaml'), 'r') as f:
    config_data = yaml.safe_load(f)

temp_path = os.path.join( _ROOT_PATH, "temp")
if not os.path.exists(temp_path):
    os.mkdir(temp_path)


def inferencer( offline_online, model_path, prompt_list_from_path, predict_list_to_path, gpu_list_str, sample_little):
    gpu_list = get_gpu_list( gpu_list_str, must=True ) if gpu_list_str!=None else []    # GPU列表（int）
    file_output_path = os.path.join( _ROOT_PATH, 'temp')                                # 临时输出路径
    if offline_online == "offline":
        "1. 离线模式：加载、推理本地prompt文件、停止"
        if gpu_list_str != None: 
            "(1) 本地模型"
            part_num = len( gpu_list )
            processes = []
            for part_id, gpu_id in enumerate( gpu_list ):
                p = multiprocessing.Process(target=Inference_offline_local, args=(part_id, part_num, gpu_id, model_path, prompt_list_from_path, file_output_path, sample_little))
                processes.append(p)
                p.start()  # 启动子进程
            # 等待所有子进程完成
            for p in processes:
                p.join()
            merge_predict( part_num, _ROOT_PATH )
            print("离线模式本地模型批预测完成！")
        else:
            "(2) 远程模型"
            Inference_offline_api( model_path, prompt_list_from_path, file_output_path, config_data, sample_little=sample_little )
            
    else:
        "2. 在线模式：启动推理引擎，bash中给出接口、端口、停止符"
        if gpu_list_str != None:
            "(1) 本地模型"
            if len(gpu_list_str)==1:
                gpu_id = int(gpu_list_str)
            else:
                raise Exception("Online Local模式GPU只能指定一个")
            p = multiprocessing.Process(target=Inference_online_local, args=(model_path, gpu_id))
            p.start()  # 启动子进程
            p.join()
        else:
            "(2) 远程模型"
            p = multiprocessing.Process(target=Inference_online_api, args=(model_path, config_data))
            p.start()  # 启动子进程
            p.join()
    merged_path = os.path.join( file_output_path, "predict_list.pickle")
    shutil.copy(merged_path, predict_list_to_path) 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run script with external arguments")
    parser.add_argument('--offline_online', type=str, required=True, help='offline or online')
    parser.add_argument('--prompt_list_from_path', type=str, required=True, help='提示词路径')
    parser.add_argument('--prompt_list_to_path', type=str, required=True, help='提示词路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--gpu_list', type=str, required=False, help='可用GPU列表，非空表示本地，否则用远程')
    parser.add_argument('--sample_little', type=str, required=False, help='小样本情况')
    args = parser.parse_args()
    # 获取参数值
    offline_online = args.offline_online
    model_path = args.model_path
    gpu_list_str = args.gpu_list_str if args.gpu_list_str else None
    prompt_list_from_path = args.prompt_list_from_path
    prompt_list_to_path = args.prompt_list_to_path
    sample_little = args.prompt_list_from_path if args.prompt_list_from_path else None
    """
    offline_online ="offline"
    local_api = "local"
    model_path = "/home/jiangpeiwen2/jiangpeiwen2/workspace/LLMs/ChatGLM3-6B/ZhipuAI/chatglm3-6b"
    gpu_list_str = "1,2,3,4,5"
    prompt_list_from_path = "/home/jiangpeiwen2/jiangpeiwen2/TKGT/code_example/Hybird_RAG/temp/cpl/cpl_data_cell_all_prompt_list.pickle"   
    sample_little = None
    """
    inferencer( offline_online, model_path, prompt_list_from_path, prompt_list_to_path, gpu_list_str, sample_little)
