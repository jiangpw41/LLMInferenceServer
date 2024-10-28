import argparse
import os
import sys  
import time
import sys
import subprocess
import multiprocessing

_ROOT_PATH = os.path.dirname( os.path.abspath(__file__) )
sys.path.insert( 0, _ROOT_PATH)
from utils import merge_predict, get_gpu_list
from offline.local.predict import Inference_offline_local
from online.local.llm_api_redis import Inference_online_local

temp_path = os.path.join( _ROOT_PATH, "temp")
if not os.path.exists(temp_path):
    os.mkdir(temp_path)


def inferencer( offline_online, local_api, model_path, prompt_list_from_path, gpu_list_str, sample_little):
    if offline_online == "offline":
        "离线模式：加载、推理本地文件、停止"
        gpu_list = get_gpu_list( gpu_list_str, must=True )
        file_input_path = prompt_list_from_path
        file_output_path = os.path.join( _ROOT_PATH, 'temp')
        if gpu_list != None:
            part_num = len( gpu_list )
            if local_api == "local":
                "本地模型"
                processes = []
                for part_id, gpu_id in enumerate( gpu_list ):
                    p = multiprocessing.Process(target=Inference_offline_local, args=(part_id, part_num, gpu_id, model_path, file_input_path, file_output_path, sample_little))
                    processes.append(p)
                    p.start()  # 启动子进程
                # 等待所有子进程完成
                for p in processes:
                    p.join()
            merge_predict( part_num, _ROOT_PATH )
            print("离线模式本地模型批预测完成！")
        else:
            "远程模型"
            raise Exception( "离线模式的API服务尚未开发" )
    else:
        "在线模式：启动推理引擎，bash中给出接口、端口、停止符"
        if local_api=="local":
            if len(gpu_list_str)==1:
                gpu_id = int(gpu_list_str)
            else:
                raise Exception("Online Local模式GPU只能指定一个")
            p = multiprocessing.Process(target=Inference_online_local, args=(model_path, gpu_id))
            p.start()  # 启动子进程
            p.join()
        else:
            raise Exception( "在线远程模式尚未开发" )

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run script with external arguments")
    parser.add_argument('--offline_online', type=str, required=True, help='offline or online')
    parser.add_argument('--local_api', type=str, required=True, help='local model or api')
    parser.add_argument('--prompt_list_from_path', type=str, required=False, help='提示词路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--gpu_list', type=str, required=True, help='可用GPU列表')
    args = parser.parse_args()
    # 获取参数值
    offline_online = args.offline_online
    local_api = args.local_api
    model_path = args.model_path
    gpu_list_str = args.gpu_list
    prompt_list_from_path = args.prompt_list_from_path
    sample_little = 0
    inferencer( offline_online, local_api, model_path, prompt_list_from_path, gpu_list_str, sample_little)
