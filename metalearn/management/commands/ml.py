import redis
import glob
import os
import json
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from metalearn.models import Architecture
from metalearn.models import Environment
from metalearn.models import Optimiser
from metalearn.models import Episode

from metalearn.ml.architectures import default_models as default_architectures
from metalearn.ml.environments import default_models as default_environments
from metalearn.ml.optimisers import default_models as default_optimisers

helptext = '''
    manage.py ml loaddefaults

    manage.py ml storage status

    manage.py ml storage sync [delete-fs] [delete-db]
        Optional 'delete-db' -> Delete database entrys missing from Filesystem
        Optional 'delete-fs' -> Delete filesystem data thats missing from  Database
'''

class Command(BaseCommand):
    help = helptext
    

    def add_arguments(self, parser):
        parser.add_argument('action', nargs='?')
        parser.add_argument('argument0', nargs='?')
        parser.add_argument('argument1', nargs='?')
        parser.add_argument('argument2', nargs='?')

    def handle(self, *args, **options):
        if options["action"] == "loaddefaults":
            for defaults,_class in [[default_architectures, Architecture],[default_environments, Environment],[default_optimisers, Optimiser]]:
                for default_architecture in defaults: 
                    a = _class()
                    a.name = default_architecture["name"]
                    a.description = default_architecture["description"]
                    a.classname = default_architecture["classname"]
                    a.classargs = json.dumps(default_architecture["classargs"])
                    try:
                        a.save()
                    except: 
                        print("Already exists: %s \tArgs: %s" % (a.classname, a.classargs))

        elif options["action"] == "storage" and options["argument0"] != None :
            self.handle_storage(*args, **options)
        elif options["action"] == "download":
            self.handle_download()
        else:
            print("Unknown options")
            print(helptext)


    def handle_download(self):
        if not os.path.isfile("metalearn/ml/datasets/mnist_train.csv"):
            os.system("cd metalearn/ml/datasets ; wget 'https://pjreddie.com/media/files/mnist_train.csv'")
        if not os.path.isfile("metalearn/ml/datasets/mnist_test.csv"):
            os.system("cd metalearn/ml/datasets ; wget 'https://pjreddie.com/media/files/mnist_test.csv'")


    def handle_storage(self, *args, **options):
        
        paths_fs = set(glob.glob("metalearn/ml/data/set_*/exp_*/ep_*/"))
        paths_db_hasFolder = []
        path_to_id = {}
        print(paths_fs)
        print(paths_db_hasFolder)
        with connection.cursor() as cursor:
            query = 'SELECT experimentSet_id as a, experiment_id as b, version, id as c FROM metalearn_Episode WHERE hasFolder = 1 GROUP BY a,b,c'
            #print(query)
            cursor.execute(query)
            rows = cursor.fetchall()
            for row in rows:
                path_to_id["metalearn/ml/data/set_%s/exp_%s/ep_%s/" % (row[0], row[1], row[2])] = row[3]
                paths_db_hasFolder.append("metalearn/ml/data/set_%s/exp_%s/ep_%s/" % (row[0], row[1], row[2]))
        paths_db_hasFolder = set(paths_db_hasFolder)
        missing_in_fs = paths_db_hasFolder - paths_fs
        should_not_be_in_fs = paths_fs - paths_db_hasFolder

        if options["argument0"] in ["status", "sync"]:
            print("Storage Status:")

            print(" %s Folders existing in Database" % len(paths_db_hasFolder))
            print(" %s Folders existing in Filesystem" % len(paths_fs))

            print(" %s Abandoned Folders existing in Filesystem that don't exist in db" % len(should_not_be_in_fs))
            if len(should_not_be_in_fs) > 0:
                print("   Abandoned Folders:")
                for name in should_not_be_in_fs:
                    print("     %s" % name)  

            print(" %s Folders missing in Filesystem but referenced in db" % len(missing_in_fs))
            if len(missing_in_fs) > 0:
                print("   Missing Folders:")
                for name in should_not_be_in_fs:
                    print("     %s" % name)  
            print("")

        if options["argument0"] in ["sync"]:
            print("Storage Sync")


            if len(missing_in_fs) == 0 and len(should_not_be_in_fs) == 0:
                print(" No changes to sync")

            else:

                if len(missing_in_fs) > 0:
                    ids = [ path_to_id[p] for p in missing_in_fs]
                    items = _class.objects.filter(id__in=ids)

                    for item in items:
                        if "delete-db" in [options["argument1"], options["argument2"]]:
                            print("  Remove Episode '%s' from database" % item.id)
                            item.delete()
                        else:
                            print("  Mark Episodes '%s' to have no folder in database" % item.id)
                            item.hasFolder = False
                            item.save()

                if len(should_not_be_in_fs) > 0:
                    if "delete-fs" in [options["argument1"], options["argument2"]]:
                        for to_delete in should_not_be_in_fs:
                            print("  Deleting '%s' from fs" % to_delete  )
                            os.system("rm -r '%s'" % to_delete  )
                        os.system("find 'metalearn/ml/data/' -type d -empty -delete")
                    else:
                        print(" %s items should not be in Filesystem, use 'delete-fs' option to force deletion" % (len(should_not_be_in_fs)))
                
            print("")




