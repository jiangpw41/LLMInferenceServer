import os
import pickle

def get_gpu_list( gpu_list_str, must=True ):
    if gpu_list_str != None:
        return list(map(int, gpu_list_str.split(',')))
    else:
        print( "未指定GPU，默认为0号")
        return [0]

def merge_predict( part_num, _ROOT_PATH ):
    ret = []
    file_output_path = os.path.join( _ROOT_PATH, 'temp')
    for i in range(part_num):
        with open( os.path.join( file_output_path, str(i)), 'rb') as f:  
            part_data = pickle.load(f)
        ret.extend( part_data )
    save_path = os.path.join( file_output_path, "predict_list.pickle")
    with open( save_path, 'wb') as f:  
        pickle.dump(ret, f)
    
    
    

