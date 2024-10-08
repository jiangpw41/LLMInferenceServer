import os

import pickle

def get_gpu_list( gpu_list_str, must=True ):
    if gpu_list_str:
        return list(map(int, gpu_list_str.split(',')))
    else:
        if must:
            raise Exception("Please Specify GPUs!")
        else:
            return None

def merge_predict( part_num, _ROOT_PATH ):
    ret = []
    file_output_path = os.path.join( _ROOT_PATH, 'offline/temp')
    for i in range(part_num):
        with open( os.path.join( file_output_path, str(i)), 'rb') as f:  
            part_data = pickle.load(f)
        ret.extend( part_data )
    with open( os.path.join( file_output_path, "predict_list.pickle"), 'wb') as f:  
        pickle.dump(ret, f)

