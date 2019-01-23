import redis
import glob
import os
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from metalearn.models import Architecture
from metalearn.models import Environment
from metalearn.models import Optimiser
from metalearn.models import Episode

from metalearn.ml.architectures import all_architectures
from metalearn.ml.environments import all_environments
from metalearn.ml.optimisers import all_optimisers

helptext = '''
    manage.py ml opti[miser] status 
        Show status of Optimisers

    manage.py ml opti[miser] sync [delete]
        Optional 'delete' -> Delete database entrys missing from Filesystem

    manage.py ml env[ironment] status 
        Show status of Environments

    manage.py ml env[ironment] sync [delete]
        Optional 'delete' -> Delete database entrys missing from Filesystem

    manage.py ml arch[itecture] status 
        Show status of Architectures

    manage.py ml arch[itecture] sync [delete]
        Optional 'delete' -> Delete database entrys missing from Filesystem

    manage.py ml storage status 
        Show status of Storage

    manage.py ml storage sync [delete-fs] [delete-db]
        Optional 'delete-db' -> Delete database entrys missing from Filesystem
        Optional 'delete-fs' -> Delete filesystem data thats missing from  Database
'''

class Command(BaseCommand):
    help = helptext
    

    def add_arguments(self, parser):
        parser.add_argument('scope', nargs='?')
        parser.add_argument('action', nargs='?')
        parser.add_argument('argument0', nargs='?')
        parser.add_argument('argument1', nargs='?')


    def handle(self, *args, **options):
        print()
        if  options["scope"] in [ "architecture", "arch", "environment", "env", "optimiser", "opti"] and  options["action"] in ["status", "sync"]:
            self.handle_aeo(*args, **options)
        elif options["scope"] in [ "storage"] and  options["action"] in ["status", "sync"]:
            self.handle_storage(*args, **options)
        else:
            print(helptext)

    def handle_storage(self, *args, **options):
        
        paths_fs = set(glob.glob("metalearn/ml/data/set_*/exp_*/ep_*/"))
        paths_db_hasFolder = []
        path_to_id = {}
        #print(paths_fs)
        #print(paths_db_hasFolder)
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

        if options["action"] in ["status", "sync"]:
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

        if options["action"] in ["sync"]:
            print("Storage Sync")


            if len(missing_in_fs) == 0 and len(should_not_be_in_fs) == 0:
                print(" No changes to sync")

            else:

                if len(missing_in_fs) > 0:
                    ids = [ path_to_id[p] for p in missing_in_fs]
                    items = _class.objects.filter(id__in=ids)

                    for item in items:
                        if "delete-db" in [options["argument0"], options["argument1"]]:
                            print("  Remove Episode '%s' from database" % item.id)
                            item.delete()
                        else:
                            print("  Mark Episodes '%s' to have no folder in database" % item.id)
                            item.hasFolder = False
                            item.save()

                if len(should_not_be_in_fs) > 0:
                    if "delete-fs" in [options["argument0"], options["argument1"]]:
                        for to_delete in should_not_be_in_fs:
                            print("  Deleting '%s' from fs" % to_delete  )
                            os.system("rm -r '%s'" % to_delete  )
                        os.system("find 'metalearn/ml/data/' -type d -empty -delete")
                    else:
                        print(" %s items should not be in Filesystem, use 'delete-fs' option to force deletion" % (len(should_not_be_in_fs)))
                
            print("")



    def handle_aeo(self, *args, **options):

        if options["scope"] in [ "architecture", "arch"]:   
            classname = "Architecture"
            _class = Architecture
            definitions = all_architectures
        if options["scope"] in [ "environment", "env" ]:
            classname = "Environment"
            _class = Environment
            definitions = all_environments
        if options["scope"] in [ "optimiser", "opti"] :
            classname = "Optimiser"
            _class = Optimiser
            definitions = all_optimisers

        if options["action"] in ["status", "sync"]:            
            db_objects = set(_class.objects.all().values_list('name', flat=True))
            fs_objects = set([ "%s" % k for k in definitions.keys()])

            missing_in_db = fs_objects - db_objects
            missing_in_fs = db_objects - fs_objects
            missing_in_fs_broken_in_db = set(_class.objects.filter(name__in=missing_in_fs, broken=True).values_list('name', flat=True))
            missing_in_fs_not_broken_in_db = missing_in_fs - missing_in_fs_broken_in_db


            print("%s Status:" % classname)
            print(" %s %s existing in Database" % ( len(db_objects)    , classname) )
            print(" %s %s missing in Database"  % ( len(missing_in_db) , classname) )
            if len(missing_in_db) > 0:
                print("   %s missing in Database:" % classname)
                for name in missing_in_db:
                    print("     %s" % name)

            print(" %s %s existing on Filesystem" % ( len(fs_objects)    , classname) )
            print(" %s %s missing on Filesystem"  % ( len(missing_in_fs) , classname) )
            if len(missing_in_fs_broken_in_db) > 0:
                print("   %s %s missing on Filesystem, marked broken in db" % ( len(missing_in_fs_broken_in_db), classname))
                for name in missing_in_fs_broken_in_db:
                    print("     %s" % name)

            if len(missing_in_fs_not_broken_in_db) > 0:
                print("   %s %s missing on Filesystem, not marked broken in db" % ( len(missing_in_fs_not_broken_in_db), classname))
                for name in missing_in_fs_not_broken_in_db:
                    print("     %s" % name)
            print("")

        if options["action"] == "sync":
            print("%s Sync:" % classname)

            if ( len(missing_in_db) > 0 or len(missing_in_fs) > 0):
                for name in missing_in_db:
                    print(" Create %s '%s' in database" % (classname, name))
                    item = _class()
                    item.name = name
                    item.description = definitions[name]["description"]
                    item.save()
                
                if options["argument0"] == "delete":
                    items = _class.objects.filter(name__in=missing_in_fs)
                    for item in items:
                        print(" %s '%s' deleted from database" % (classname, item.name))
                        item.delete()

                else:
                    items = _class.objects.filter(name__in=missing_in_fs, broken=False)
                    for item in items:
                        print(" Mark %s '%s' as broken in database" % (classname, item.name))
                        item.broken = True
                        item.save()
            else:
                print(" No %s to sync" % classname)
            print("")
                    














