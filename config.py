import os

# Config file, put all your keys and passwords and whatnot in here
DB_URI = os.environ.get("DATABASE_URL", "sqlite:///database.db")
SECRET_KEY = "hackbright"
CSRF_ENABLED = True
