import sqlite3

conn = sqlite3.connect('FaceTestDb')
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    image BLOB NOT NULL
)
""")
print("✅ Table created.")
conn.commit()
conn.close()
