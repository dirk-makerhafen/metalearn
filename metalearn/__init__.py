from django.db.backends.signals import connection_created
from .celery import app as celery_app


def enabled_pragmas(sender, connection, **kwargs):
    """Enable integrity constraint with sqlite."""
    if connection.vendor == 'sqlite':
        cursor = connection.cursor()
        cursor.execute('PRAGMA foreign_keys = ON;')
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = OFF;")
        cursor.execute("PRAGMA cache_size = 512000;")
        cursor.execute('PRAGMA temp_store = MEMORY;')
        
connection_created.connect(enabled_pragmas)


__all__ = ('celery_app',)