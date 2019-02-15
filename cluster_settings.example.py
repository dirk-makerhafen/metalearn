
git_url = "https://github.com/dirk-attraktor/metalearn.git"

default_remote_dir = "/tmp/metalearn"

# the url used for master at via 'python3 manage.py runserver 1.2.3.4:1234'
master_url = ""

# password == None      -> use ssh keys
# password == ""        -> ask for password (cached im process memory)
# password == "somepassword" -> use "somepassword" 
# pre -> some script to be executed before every command

cluster_nodes = [
    { "name": "somename", "host": "1.2.3.4", "port": "22", "username": "user", "password": None, "remote_dir": default_remote_dir , "pre": ""},
]
