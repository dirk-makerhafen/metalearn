import os
import json
import requests
import numpy
import random
import io
import time
import redis 
import glob

from metalearn.ml import environments
from metalearn.ml import architectures
from metalearn.ml import optimisers

redisconnection = redis.StrictRedis()

MAXEXECUTIONS = sys.argv[1]
APIURL = sys.argv[2]

def createNoise(seed, width):
    r = numpy.random.RandomState(seed)
    return r.randn(width).astype(numpy.float32)

class WeightsNoiseCache():
    def __init__(self):
        os.system("mkdir /tmp/WeightsNoiseCache/ 2>/dev/null")
        files = glob.glob("/tmp/WeightsNoiseCache/*.cache")
        for f in files: # refresh db just in case its gone
            episode_id = f.split("/")[-1].split(".")[0]
            redisconnection.zadd("WeightsNoiseCache_ids", { episode_id : float(time.time())}, ex=1200 )
            
    def get(self, episode_id):
        fname = "/tmp/WeightsNoiseCache/%s.cache" % episode_id
        if os.path.isfile(fname) == True:
            redisconnection.zadd("WeightsNoiseCache_ids", { episode_id : float(time.time())}, ex=1200 )
            print("Episode '%s' WeightsNoise via fs cache" % episode_id)
            return numpy.load(io.BytesIO(open(fname,"rb").read()))

        r = redisconnection.set("episode_weightsNoise_%s_dlactive" % episode_id, 'true', ex=1200, nx=True)
        if r == True:
            print("Download Episode WeightsNoise '%s' via web" % episode_id)
            try:
                weightsNoiseData = requests.get("%s/getEpisodeWeightNoise/%s" % ( APIURL, episode_id ) ).content
            except Exception as e:
                print(e)    
                return None
            open(fname,"wb").write(weightsNoiseData)
            redisconnection.delete("episode_weightsNoise_%s_dlactive" % episode_id)

        r = redisconnection.get("episode_weightsNoise_%s_dlactive" % episode_id) # evtl. some other theads does this dl
        while r != None:
            print("waiting for download of episode '%s' in other thread: %s" % (episode_id, r))
            time.sleep(2)
            r = redisconnection.get("episode_weightsNoise_%s_dlactive" % episode_id) # evtl. some other theads does this dl
                
        redisconnection.zadd("WeightsNoiseCache_ids",{ episode_id : float(time.time())}, ex=1200 )

        cached_ids = redisconnection.zrevrange("WeightsNoiseCache_ids", 0, -1, withscores=True)
        print(cached_ids)
        if len(cached_ids) > 20:
            print("Cache to large")
            for cached_id in cached_ids[20:]:
                removed = redisconnection.zrem("WeightsNoiseCache_ids", cached_id[0])
                if removed > 0:
                    print("removed")
                    os.system("rm '/tmp/WeightsNoiseCache/%s.cache'" % cached_id[0])
                
        return numpy.load(io.BytesIO(open(fname,"rb").read()))        

    def getCachedIds(self):
        cached_ids = redisconnection.zrevrange("WeightsNoiseCache_ids", 0, -1)
        r = [int(k) for k in cached_ids]
        print(r)
        return r

weightsNoiseCache = WeightsNoiseCache()


def getNextEpisodeNoisyExecution():
    noisyExecution = requests.get("%s/getEpisodeNoisyExecution/%s" % (APIURL, ",".join(["%s"%x for x in weightsNoiseCache.getCachedIds()]))).text
    return json.loads(noisyExecution)
    
def getEnvironmentInstance(name):
    return environments.all_environments[name]["class"]()

def getArchitectureInstance(name):
    return architectures.all_architectures[name]["class"]()

def run(nr_of_executions = 1):
    
    # for speedup so we don't have to initialize every time
    last_environment = None
    last_architecture = None
    last_envarchkey = ""

    for _ in range(nr_of_executions):
        noisyExecution = getNextEpisodeNoisyExecution()
        if noisyExecution == None:
            print("Nothing to do, waiting")
            time.sleep(10)
            continue

        start = time.time()

        weightsNoise = weightsNoiseCache.get(noisyExecution["episode.id"])  # [0] -> Weights , [1] -> NoiseLevels
        weights_new = weightsNoise[0] + (weightsNoise[1] * createNoise(noisyExecution["noiseseed"], len(weightsNoise[0] ) ) )

        weightsNoise = None # free memory 

        envarchkey = noisyExecution["environment.name"] + "__" + noisyExecution["architecture.name"]
        if envarchkey == last_envarchkey: # use last used arch/env
            print("Using env/arch from cache")
            environment = last_environment
            environment.reset()
            architecture = last_architecture
            architecture.reset(weights_new)
        else:
            print("Using NEW env/arch")
                        
            if last_environment != None:
                last_environment.close()
            if last_architecture != None:
                last_architecture.close()
            environment = getEnvironmentInstance(noisyExecution["environment.name"])
            architecture = getArchitectureInstance(noisyExecution["architecture.name"])
            environment.initialize()
            architecture.initialize(environment, weights_new)
        
        weights_new = None # free memory 
        last_architecture = architecture
        last_environment  = environment
        last_envarchkey = envarchkey
        fitness = 0
        count = 0

        while environment.hasNextObservation():
            observation = environment.getNextObservation()
            action = architecture.run(observation)
            fitness += environment.runAction(action) 
            #env.env.render()
            count += 1
            if count > 10000:
                break
        stop = time.time()
        
        results = json.dumps({
            "fitness" : fitness,
            "steps" : count,
            "timespend" : int(stop - start),
        })
        print(results)
        requests.post("%s/putResult/%s/%s" % (APIURL, noisyExecution["id"], noisyExecution["lock"]), results)

    if last_environment != None:
        last_environment.close()
    if last_architecture != None:
        last_architecture.close()     
  
run(MAXEXECUTIONS)

