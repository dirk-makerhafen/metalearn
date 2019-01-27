import os
import json
import requests
import numpy
import random
import io
import time
import redis 
import glob
import sys
import threading
import multiprocessing
import signal


from metalearn.ml import environments
from metalearn.ml import architectures
from metalearn.ml import optimisers

try:
    redisconnection = redis.StrictRedis(unix_socket_path='/var/run/redis/redis.sock', db=8)
    redisconnection.get("__test")
except:
    redisconnection = redis.StrictRedis(db=8)




try:
    ARG = sys.argv[1]
    if ARG in ["daemon", "run"]:
        URL = sys.argv[2]
        CNT  = int(sys.argv[3])

        if not URL.lower().startswith("http"):
            URL = 'http://%s' % URL

        if ARG == "daemon" and CNT == 0:
            CNT = multiprocessing.cpu_count()
except:
    print('''
        client.py daemon url cores                  # run as daemon
        client.py run    url nr_of_executions       # called by daemon
        client.py stop                              # stop all active clients
        client.py status                            # show nr of active daemon/workers
    ''')
    exit(1)

ACTIVE = True
threads = []

def createNoise(seed, width):
    r = numpy.random.RandomState(seed)
    return r.randn(width).astype(numpy.float32)

class WeightsNoiseCache():
    def __init__(self):
        os.system("mkdir /tmp/WeightsNoiseCache/ 2>/dev/null")
        files = glob.glob("/tmp/WeightsNoiseCache/*.cache")

        for f in files: # refresh db just in case its gone
            episode_id = int(f.split("/")[-1].split(".")[0])
            if redisconnection.zrank("WeightsNoiseCache_ids", episode_id ) == None:
                print("Cached File '%s' no longed exists in redis, deleting" % f)
                os.system("rm '/tmp/WeightsNoiseCache/%s.cache'" % episode_id)
    
    def get(self, episode_id):
        fname = "/tmp/WeightsNoiseCache/%s.cache" % episode_id
        if os.path.isfile(fname) == True:
            redisconnection.zadd("WeightsNoiseCache_ids", { episode_id : float(time.time()) } )
            print("Episode '%s' WeightsNoise via fs cache" % episode_id)
            return numpy.load(io.BytesIO(open(fname,"rb").read()))

        r = redisconnection.set("episode_weightsNoise_%s_dlactive" % episode_id, 'true', ex=1200, nx=True)
        if r == True:
            print("Download Episode WeightsNoise '%s' via web" % episode_id)
            try:
                weightsNoiseData = requests.get("%s/getEpisodeWeightNoise/%s" % ( URL, episode_id ) ).content
            except Exception as e:
                print(e)    
                return None
            f = open(fname,"wb")
            f.write(weightsNoiseData)
            f.flush()
            f.close()
            redisconnection.delete("episode_weightsNoise_%s_dlactive" % episode_id)

        r = redisconnection.get("episode_weightsNoise_%s_dlactive" % episode_id) # evtl. some other theads does this dl
        while r != None:
            print("waiting for download of episode '%s' in other thread: %s" % (episode_id, r))
            time.sleep(2)
            r = redisconnection.get("episode_weightsNoise_%s_dlactive" % episode_id) # evtl. some other theads does this dl
                
        redisconnection.zadd("WeightsNoiseCache_ids",{ episode_id : float(time.time())})

        cached_ids = redisconnection.zrevrange("WeightsNoiseCache_ids", 0, -1, withscores=True)
        if len(cached_ids) > 20:
            print("Cache to large")
            for cached_id in cached_ids[20:]:
                removed = redisconnection.zrem("WeightsNoiseCache_ids", int(cached_id[0]))
                if removed > 0:
                    print("removed")
                    os.system("rm '/tmp/WeightsNoiseCache/%s.cache'" % int(cached_id[0]))
                
        return numpy.load(io.BytesIO(open(fname,"rb").read()))        

    def getCachedIds(self):
        cached_ids = redisconnection.zrevrange("WeightsNoiseCache_ids", 0, -1)
        r = [int(k) for k in cached_ids]
        #print(r)
        return r

weightsNoiseCache = WeightsNoiseCache()

def getNextEpisodeNoisyExecution():
    try:
        noisyExecution = requests.get("%s/getEpisodeNoisyExecution/%s" % (URL, ",".join(["%s"%x for x in weightsNoiseCache.getCachedIds()]))).text
        r = json.loads(noisyExecution)
        return r
    except Exception as e:
        print("Failed to receive next: %s " % e)
    return None
    

def getEnvironmentInstance(name):
    return environments.all_environments[name]["class"]()

def getArchitectureInstance(name):
    return architectures.all_architectures[name]["class"]()

def run():
    global CNT

    # for speedup so we don't have to initialize every time
    last_environment = None
    last_architecture = None
    last_envarchkey = ""

    while CNT > 0:
        
        noisyExecution = getNextEpisodeNoisyExecution()
        if noisyExecution == None:
            print("Nothing to do, waiting")
            time.sleep(5)
            continue

        start = time.time()

        weightsNoise = weightsNoiseCache.get(noisyExecution["episode.id"])  # [0] -> Weights , [1] -> NoiseLevels
        weights_new = weightsNoise[0] + (weightsNoise[1] * createNoise(noisyExecution["noiseseed"], len(weightsNoise[0] ) ) )

        weightsNoise = None # speedup memory free

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
            architecture.initialize(environment.observation_space, environment.action_space, weights_new)
        
        weights_new = None # free memory 
        last_architecture = architecture
        last_environment  = environment
        last_envarchkey = envarchkey
        fitness = 0
        steps = 0

        while environment.hasNextObservation():
            observation = environment.getNextObservation()
            action = architecture.run(observation)
            fitness += environment.runAction(action) 
            #env.env.render()
            steps += 1
            if steps >= noisyExecution["max_steps"]:
                break
            if int(time.time() - start) >= noisyExecution["max_timespend"]:
                break
        
        ts =  int(time.time() - start)
        results = json.dumps({
            "fitness" : fitness,
            "steps" : steps,
            "timespend" : ts,
        })
        print("%s  |  %s  | Steps: %s \tTime: %s \tFitness: %s" % (noisyExecution["environment.name"], noisyExecution["architecture.name"],  steps, ts, fitness))

        while True:
            try:
                requests.post("%s/putResult/%s/%s" % (URL, noisyExecution["id"], noisyExecution["lock"]), results)
                break
            except:
                print("Failed to post result, trying again")
                time.sleep(5)
            if CNT == 0:    
                break
        CNT -= 1

    if last_environment != None:
        last_environment.close()
    if last_architecture != None:
        last_architecture.close()     


def daemonThread(): 
    global ACTIVE
    print("Starting Thread")
    while ACTIVE == True:
        os.system("nice -n 5 python3 client.py run '%s' 200" % URL)
    print("Exiting Thread")

def stop():
    global ACTIVE
    ACTIVE = False
    print("Stoping now, this may take some time to wait for jobs to finish")
    for t in threads:
        t.join()
    print("All stopped, you may exit now")

def start():
    for _ in range(0, CNT):
        t = threading.Thread(target=daemonThread)
        t.start()
        threads.append(t)


def on_sigterm(signum, frame):
    global CNT
    global ACTIVE
    global ARG
    print("Received SIGTERM/SIGINT")
    CNT = 0
    ACTIVE = False
    if ARG == "daemon":
        os.system("ps x | grep 'python3 client.py run ' | grep -v grep | grep -v nice | cut -dp -f1 | xargs kill")
        stop()
        exit(0)


signal.signal(signal.SIGINT, on_sigterm)
signal.signal(signal.SIGTERM, on_sigterm)


if ARG == "daemon":
    print("use 'stop()' to stop and exit.")
    start()

elif ARG == "run":
    run()
elif ARG == "status":
    print("Daemon Active:")
    os.system("ps x | grep 'python3 client.py daemon ' | grep -v grep | grep -v nice | wc -l")
    print("Workers Active:")
    os.system("ps x | grep 'python3 client.py run '    | grep -v grep | grep -v nice | wc -l")


elif ARG == "stop":
    os.system("ps x | grep 'python3 client.py daemon ' | grep -v grep | grep -v nice | cut -dp -f1 | cut -d? -f1| xargs kill")
    os.system("ps x | grep 'python3 client.py run '    | grep -v grep | grep -v nice | cut -dp -f1 | cut -d? -f1| xargs kill")

else:
    print("unknown arg '%s'" % ARG)
    exit(1)

 

