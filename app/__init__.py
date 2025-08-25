from flask import Flask
from celery import Celery
from .config import Config
from .celery_config import broker_url, result_backend

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=broker_url,
        backend=result_backend
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

app = Flask(__name__)
app.config.from_object(Config)
celery = make_celery(app)
