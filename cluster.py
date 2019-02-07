import os
import sys
import getpass

from cluster_settings import *


class remote_scripts():

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

    status = '''
        if [ -d "%(remote_dir)s" ]; then
          cd "%(remote_dir)s"
          source venv/bin/activate
          python3 client.py status
        fi
    '''


    sync = '''
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

        python3 manage.py makemigrations
        python3 manage.py makemigrations metalearn

        python3 manage.py migrate
        python3 manage.py migrate metalearn
        
        python3 manage.py ml loaddefaults

        python3 manage.py ml storage sync

    '''


def remotecmd(shellscript, cluster_node):

    shellscript = shellscript % { 
        "remote_dir": cluster_node["remote_dir"],
        "master_url": master_url,
        "git_url": git_url,
    }

    if cluster_node["password"] == "":
        prompt = 'Enter ssh password for %s@%s:%s\n' % (cluster_node["username"], cluster_node["host"], cluster_node["port"] )
        cluster_node["password"] = getpass.getpass(prompt)       

    if cluster_node["password"] == None:
        sshcmd = ""
    else:
        sshcmd = "sshpass -p '%s' "  % cluster_node["password"]
    
    outputfile = "/tmp/tmp_123"

    sshcmd += "ssh -p %s %s@%s 1>'%s' 2>&1 << EOF\n" % ( cluster_node["port"], cluster_node["username"], cluster_node["host"] , outputfile)
    sshcmd += cluster_node["pre"] 
    sshcmd += "\necho __FIRSTLINE__\n"
    sshcmd += shellscript 
    sshcmd += "\necho __LASTLINE__\n"
    sshcmd += "EOF\n" 
    os.system(sshcmd)

    output = None
    if os.path.isfile(outputfile):
        output = open(outputfile,"r").read().split("__LASTLINE__")[0].split("__FIRSTLINE__")[-1] #bad
        print("HOST: %s@%s" % (cluster_node["username"], cluster_node["host"] ))
        print(output)
    else:
        print("No output generated")
    return output


try:
    ARG = sys.argv[1]
except:
    print(''' 
        cluster.py status
        cluster.py start
        cluster.py stop
        cluster.py sync
    ''')
    exit(1)


if ARG == "status":
    for cluster_node in cluster_nodes:
        remotecmd(remote_scripts.status, cluster_node)

elif ARG == "start":
    for cluster_node in cluster_nodes:
        remotecmd(remote_scripts.start, cluster_node)

elif ARG == "stop":
    for cluster_node in cluster_nodes:
        remotecmd(remote_scripts.stop, cluster_node)

elif ARG == "sync":
    for cluster_node in cluster_nodes:
        remotecmd(remote_scripts.sync, cluster_node)

else:
    print("arg '%s' unknown" % ARG)

