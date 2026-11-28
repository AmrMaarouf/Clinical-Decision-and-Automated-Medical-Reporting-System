import os

# tests always use an in-memory SQLite database, never the real Postgres
os.environ["DATABASE_URL"] = "sqlite://"