class Config:
    DEBUG = True
    broker_url = 'redis://localhost:6379'
    result_backend = 'redis://localhost:6379'
    # Add any other configurations here
