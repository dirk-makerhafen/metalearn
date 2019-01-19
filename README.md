# metalearn
MetaLearning stuff

python3 -m venv venv

source venv/bin/activate 

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 manage.py makemigrations

python3 manage.py migrate

python3 manage.py ml env sync
python3 manage.py ml arch sync
python3 manage.py ml opti sync
python3 manage.py ml storage sync

python3 manage.py runserver 1.2.3.4:1234

celery -A metalearn worker -l info

python3 client.py 4  1.2.3.4:1234

