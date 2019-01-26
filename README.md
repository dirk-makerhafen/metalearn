# metalearn
MetaLearning stuff


apt-get install python3 python3-venv python3-dev cmake zlib1g-dev

git clone https://github.com/dirk-attraktor/metalearn.git

cd metalearn

python3 -m venv venv

source venv/bin/activate 

python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt


python3 manage.py makemigrations

python3 manage.py makemigrations metalearn

python3 manage.py migrate 

python3 manage.py migrate metalearn


python3 manage.py ml env sync

python3 manage.py ml arch sync

python3 manage.py ml opti sync

python3 manage.py ml storage sync


Server:

python3 manage.py runserver 1.2.3.4:1234

celery -A metalearn worker -l info


Client:

python3 client.py daemon 1.2.3.4:1234 0  # mode, url, cores
    # 0 cores = auto
python3 client.py run    1.2.3.4:1234 1  # mode, url, nr_of_execution 


