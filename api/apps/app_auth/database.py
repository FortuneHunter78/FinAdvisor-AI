import sqlite3


conn = sqlite3.connect("auth.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS auth_keys (
        auth_key TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
        )""")

conn.commit()


