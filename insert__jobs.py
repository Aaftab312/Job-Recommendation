import pandas as pd
import sqlite3

df = pd.read_csv("jobs_dataset_1000.csv")

conn = sqlite3.connect("job.db")
cursor = conn.cursor()

# Optional: clear old data
cursor.execute("DELETE FROM jobs")

for _, row in df.iterrows():
    cursor.execute("""
    INSERT INTO jobs (job_title, skills, description)
    VALUES (?, ?, ?)
    """, (
        str(row["job_title"]),
        str(row["skills_required"]),  # CSV column
        str(row["description"])
    ))

conn.commit()
conn.close()

print("Jobs inserted successfully.")

