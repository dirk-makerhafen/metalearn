import os
import sys
import getpass
import threading
import queue
import pprint

from cluster_settings import *


taskqueue = queue.Queue()

class remote_scripts():

    status = '''
        if [ -d "%(remote_dir)s" ]; then
          cd "%(remote_dir)s"
          source venv/bin/activate
          python3 client.py status
        fi
    '''

    start = '''
        if [ -d "%(remote_dir)s" ]; then
          cd %(remote_dir)s
          source venv/bin/activate
          echo starting daemon
          nohup python3 client.py daemon "%(master_url)s" 0   1>/dev/null 2>&1 & 
        fi
    '''

    stop = '''
        if [ -d "%(remote_dir)s" ]; then
          cd %(remote_dir)s
          source venv/bin/activate
          python3 client.py stop
        fi
    '''

    pull = '''
        if [ ! -d "%(remote_dir)s" ]; then
          mkdir -p "%(remote_dir)s"
          cd "%(remote_dir)s"
          git clone "%(git_url)s" .
        fi

        cd "%(remote_dir)s"

        git pull

        python3 -m venv venv
        source venv/bin/activate

        python3 -m pip install --upgrade pip
        python3 -m pip install -r requirements.txt

        rm db.sqlite*
        rm -r metalearn/migrations
        rm -r /tmp/WeightsNoiseCache/*.cache

        python3 manage.py makemigrations
        python3 manage.py makemigrations metalearn

        python3 manage.py migrate
        python3 manage.py migrate metalearn
        
        python3 manage.py ml loaddefaults

        python3 manage.py ml storage sync
    '''

    buildtf = '''
        export DEBIAN_FRONTEND=noninteractive

        add-apt-repository ppa:webupd8team/java
        echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" > /etc/apt/sources.list.d/bazel.list
        curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
        
        apt-get update 
        apt-get -y --allow-unauthenticated install openjdk-8-jdk
        apt-get -y --allow-unauthenticated install oracle-java8-installer
        apt-get -y --allow-unauthenticated install bazel

        cd "%(remote_dir)s"

        source venv/bin/activate
        export PYTHONVERSION="\`python3 --version | sed s/'Python '// | cut -d. -f1-2\`"        
        
        python3 -m pip install pip six numpy wheel mock
        python3 -m pip install keras_applications==1.0.6 --no-deps
        python3 -m pip install keras_preprocessing==1.0.5 --no-deps

        if [ ! -d "%(remote_dir)s/tensorflow" ]; then
            git clone https://github.com/tensorflow/tensorflow.git
        fi

        cd tensorflow

        git pull

        cat > .tf_configure.bazelrc <<END
            build --action_env PYTHON_BIN_PATH="%(remote_dir)s/venv/bin/python3"
            build --action_env PYTHON_LIB_PATH="%(remote_dir)s/venv/lib/python\$PYTHONVERSION/site-packages/"
            build --python_path="%(remote_dir)s/venv/bin/python3"
            build:xla --define with_xla_support=true
            build --config=xla
            build --action_env TF_NEED_OPENCL_SYCL="0"
            build --action_env TF_NEED_ROCM="0"
            build --action_env TF_NEED_CUDA="0"
            build --action_env TF_DOWNLOAD_CLANG="0"
            build:opt --copt=-march=native
            build:opt --copt=-Wno-sign-compare
            build:opt --host_copt=-march=native
            build:opt --define with_default_optimizations=true
            build:v2 --define=tf_api_version=2
        END

        rm -r /tmp/tensorflow_pkg* 2>/dev/null
    
        bazel build --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package

        ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

        python3 -m pip install --upgrade /tmp/tensorflow_pkg/tensorflow*.whl
    '''

    clear = '''
        if [ -d "%(remote_dir)s" ]; then
          rm -rf %(remote_dir)s
        fi
    '''

    #remove leading spaced that only exist because this formating in python is nicer 
    start = "\n".join([x[8:] for x in start.split("\n")])
    stop = "\n".join([x[8:] for x in stop.split("\n")])
    clear = "\n".join([x[8:] for x in clear.split("\n")])
    status = "\n".join([x[8:] for x in status.split("\n")])
    pull = "\n".join([x[8:] for x in pull.split("\n")])
    buildtf = "\n".join([x[8:] for x in buildtf.split("\n")])

def remotecmd(shellscript, cluster_node):

    shellscript = shellscript % { 
        "remote_dir": cluster_node["remote_dir"],
        "master_url": master_url,
        "git_url": git_url,
    }

    if cluster_node["password"] == "":
        prompt = 'Enter ssh password for %s@%s:%s\n' % (cluster_node["username"], cluster_node["host"], cluster_node["port"] )
        cluster_node["password"] = getpass.getpass(prompt)       

    outputfile = "/tmp/tmp_%s" % cluster_node["host"]

    if cluster_node["password"] == None:
        sshcmd = ""
    else:
        sshcmd = "sshpass -p '%s' "  % cluster_node["password"]
    
    sshcmd += "ssh -p %s %s@%s 2>&1 << EOF|sed -e 's/^/%s /'\n" % ( cluster_node["port"], cluster_node["username"], cluster_node["host"] , cluster_node["host"])
    sshcmd += cluster_node["pre"] 
    sshcmd += "\necho '########################################'\n"
    sshcmd += shellscript 
    sshcmd += "\necho '########################################'\n"
    sshcmd += "EOF\n" 
    sshcmd += "" 
    os.system(sshcmd)

def worker():
    while True:
        task = taskqueue.get()
        remotecmd(task[0], task[1])
        taskqueue.task_done()


try:
    ARG = sys.argv[1]
except:
    print(''' 
        cluster.py show                # show cluster hosts
        cluster.py status  <host|name> # show status
        cluster.py start   <host|name> # start cluster 
        cluster.py stop    <host|name> # stop cluster
        cluster.py pull    <host|name> # deploy project from git
        cluster.py buildtf <host|name> # build tensorflow from source
        cluster.py clear   <host|name> # delete deployed install dir
    ''')
    exit(1)

try:
    HOST = sys.argv[2]
except:
    HOST = None

for _ in range(0, min([10,len(cluster_nodes)])):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

if ARG == "show":
    for cluster_node in cluster_nodes: 
        print("Name: %s,\t Host: %s,\t Port: %s,\t RemoteDir: %s" % (cluster_node["name"], cluster_node["host"], cluster_node["port"], cluster_node["remote_dir"]))    
    exit(0)

elif ARG == "status":
    rs = remote_scripts.status
elif ARG == "start":
    rs = remote_scripts.start
elif ARG == "stop":
    rs = remote_scripts.stop
elif ARG == "pull":
    rs = remote_scripts.pull
elif ARG == "buildtf":
    rs = remote_scripts.buildtf
elif ARG == "clear":
    rs = remote_scripts.clear
else:
    print("arg '%s' unknown" % ARG)
    exit(1)


for cluster_node in cluster_nodes:
    if HOST == None or HOST == cluster_node["host"] or HOST == cluster["name"]:
        taskqueue.put([rs, cluster_node])
         

taskqueue.join()
