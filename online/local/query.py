import redis
import pickle

redis_communication = redis.StrictRedis(host='localhost', port=6379, db=0)

def chat_redis( prompt, redis_communication, index=0 ):
    redis_communication.rpush('Tasks', pickle.dumps( (index, prompt) ))
    while True:
        message = redis_communication.lpop('Response')
        if message!=None:
            index, task = pickle.loads(message)
            return task