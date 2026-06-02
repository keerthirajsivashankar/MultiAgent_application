import sqlite3

# Create/connect database
conn = sqlite3.connect("data/telecom_ops.db")

# Create cursor
cursor = conn.cursor()

# Read schema SQL
with open("sql/01_schema.sql", "r") as f:
    schema_sql = f.read()

# Execute schema
cursor.executescript(schema_sql)

print("Schema created successfully.")

# Read seed SQL
with open("sql/02_seed_data.sql", "r") as f:
    seed_sql = f.read()

# Execute seed data
cursor.executescript(seed_sql)

print("Seed data inserted successfully.")

# Commit and close
conn.commit()
conn.close()

print("Database created: data/telecom_ops.db")
