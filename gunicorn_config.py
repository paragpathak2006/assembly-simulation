workers = 16
bind = "0.0.0.0:8000"
chdir = "/home/ubuntu/buildit"
wsgi_app = "app.app:app"
timeout = 86400
graceful_timeout = 120
