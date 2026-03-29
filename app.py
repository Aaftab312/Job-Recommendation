from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import sqlite3
import numpy as np
import re
import joblib
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from rapidfuzz import fuzz
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from flask_mail import Mail, Message


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# -----------------------------
# SECURITY / RESET CONFIG
# -----------------------------
app.config["SECURITY_PASSWORD_SALT"] = "your_password_salt_456"

# -----------------------------
# MAIL CONFIG
# -----------------------------
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = "your_email@gmail.com"

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

# -----------------------------
# LOAD SPACY MODEL
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# LOAD FULL MODEL FROM model.pkl
# -----------------------------
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found. Run model.py first.")

model_data = joblib.load(MODEL_PATH)

tfidf = model_data["tfidf"]
tfidf_matrix = model_data["tfidf_matrix"]
df = model_data["df"]
PHRASES = model_data["phrases"]

# safety cleanup
df["job_title"] = df["job_title"].fillna("").astype(str)
df["skills_required"] = df["skills_required"].fillna("").astype(str)
df["description"] = df["description"].fillna("").astype(str)

# -----------------------------
# SKILL EXTRACTION LISTS
# -----------------------------
BASE_SKILLS = {
    "python", "java", "sql", "machine learning", "deep learning",
    "data science", "data analysis", "data visualization",
    "natural language processing", "nlp", "tensorflow", "pytorch",
    "pandas", "numpy", "excel", "statistics", "hadoop", "big data",
    "etl", "oop", "algorithms", "data structures", "flask", "fastapi",
    "power bi", "tableau", "scikit-learn", "computer vision", "opencv",
    "html", "css", "javascript", "react", "node.js", "django", "mysql",
    "mongodb", "git", "github", "aws", "azure", "c", "c++"
}

GENERIC_SKILLS = {
    "management", "server", "servers", "system", "systems",
    "technology", "technologies", "tool", "tools",
    "application", "applications", "software", "service", "services",
    "security", "automation"
}


def build_skill_dictionary():
    dataset_skills = set()

    for value in df["skills_required"].dropna():
        value = str(value).lower()

        temp_text = value
        for phrase in PHRASES:
            if phrase in temp_text:
                dataset_skills.add(phrase)
                temp_text = temp_text.replace(phrase, " ")

        for token in temp_text.replace(",", " ").split():
            token = token.strip().lower()
            if token:
                dataset_skills.add(token)

    return sorted(BASE_SKILLS.union(dataset_skills))


ALL_SKILLS = build_skill_dictionary()

phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in ALL_SKILLS if skill]
phrase_matcher.add("SKILLS", patterns)


def get_db_connection():
    conn = sqlite3.connect("job.db")
    conn.row_factory = sqlite3.Row
    return conn


def is_logged_in():
    return "account_id" in session


def generate_reset_token(email):
    return serializer.dumps(email, salt=app.config["SECURITY_PASSWORD_SALT"])


def verify_reset_token(token, expiration=1800):
    try:
        email = serializer.loads(
            token,
            salt=app.config["SECURITY_PASSWORD_SALT"],
            max_age=expiration
        )
        return email
    except SignatureExpired:
        return None
    except BadSignature:
        return None


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#./ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills_as_phrases(text):
    text = str(text).lower()
    found = set()

    for phrase in PHRASES:
        if phrase in text:
            found.add(phrase)
            text = text.replace(phrase, " ")

    for token in text.replace(",", " ").split():
        token = token.strip()
        if token:
            found.add(token)

    return found


def extract_skill_sets(user_skills, job_skills_text):
    user_skill_set = extract_skills_as_phrases(user_skills)
    job_skill_set = extract_skills_as_phrases(job_skills_text)

    matched_skills = sorted(list(user_skill_set & job_skill_set))
    missing_skills = sorted(list(job_skill_set - user_skill_set))

    return matched_skills, missing_skills


def extract_text_from_resume(file):
    filename = file.filename.lower()

    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    if filename.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    return ""


def extract_skills_from_resume_advanced(text):
    text = normalize_text(text)

    doc = nlp(text)
    matches = phrase_matcher(doc)

    found = set()
    for _, start, end in matches:
        skill = doc[start:end].text.lower().strip()
        if skill not in GENERIC_SKILLS:
            found.add(skill)

    tokens = text.split()
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens)) + 1):
            chunk = " ".join(tokens[i:j]).strip()

            if len(chunk) < 2:
                continue

            for skill in ALL_SKILLS:
                if skill in GENERIC_SKILLS:
                    continue
                if fuzz.ratio(chunk, skill) >= 92:
                    found.add(skill)

    mapping = {
        "ml": "machine learning",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "ds": "data structures",
        "da": "data analysis"
    }

    normalized = set()
    for skill in found:
        normalized_skill = mapping.get(skill, skill)
        if normalized_skill not in GENERIC_SKILLS:
            normalized.add(normalized_skill)

    return sorted(normalized)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        conn = get_db_connection()
        cursor = conn.cursor()

        existing = cursor.execute(
            "SELECT id FROM accounts WHERE email = ?",
            (email,)
        ).fetchone()

        if existing:
            conn.close()
            flash("Email already registered.", "error")
            return render_template("signup.html")

        password_hash = generate_password_hash(password)

        cursor.execute("""
            INSERT INTO accounts (username, email, password_hash)
            VALUES (?, ?, ?)
        """, (username, email, password_hash))

        conn.commit()
        conn.close()

        flash("Signup successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        conn = get_db_connection()
        account = conn.execute(
            "SELECT * FROM accounts WHERE email = ?",
            (email,)
        ).fetchone()
        conn.close()

        if account and check_password_hash(account["password_hash"], password):
            session["account_id"] = account["id"]
            session["username"] = account["username"]
            session["email"] = account["email"]

            flash("Login successful.", "success")
            return redirect(url_for("home"))

        flash("Invalid email or password.", "error")
        return render_template("login.html")

    return render_template("login.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"].strip().lower()

        conn = get_db_connection()
        account = conn.execute(
            "SELECT * FROM accounts WHERE email = ?",
            (email,)
        ).fetchone()
        conn.close()

        if not account:
            flash("Email not found.", "error")
            return render_template("forgot_password.html")

        token = generate_reset_token(email)
        reset_link = url_for("reset_password", token=token, _external=True)

        try:
            msg = Message(
                subject="Reset Your Password - AI Job Recommendation System",
                recipients=[email]
            )

            # plain text fallback
            msg.body = f"""
Hello,

We received a request to reset your password for your AI Job Recommendation System account.

Click the link below to reset your password:
{reset_link}

This link will expire in 30 minutes.

If you did not request a password reset, please ignore this email.

Regards,
AI Job Recommendation System
"""

            # HTML email with button
            msg.html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Reset Password</title>
</head>
<body style="margin:0; padding:0; background-color:#f4f7fb; font-family:Arial, sans-serif;">
    <div style="max-width:600px; margin:40px auto; background:#ffffff; border-radius:12px; padding:30px; box-shadow:0 0 12px rgba(0,0,0,0.08);">
        
        <h2 style="color:#1f4e79; margin-top:0;">Reset Your Password</h2>

        <p style="font-size:15px; color:#333;">
            Hello,
        </p>

        <p style="font-size:15px; color:#333; line-height:1.6;">
            We received a request to reset your password for your
            <strong>AI Job Recommendation System</strong> account.
        </p>

        <p style="font-size:15px; color:#333; line-height:1.6;">
            Click the button below to reset your password:
        </p>

        <div style="text-align:center; margin:30px 0;">
            <a href="{reset_link}"
               style="background-color:#1f4e79; color:#ffffff; text-decoration:none; padding:14px 28px; border-radius:8px; display:inline-block; font-size:16px; font-weight:bold;">
               Reset Password
            </a>
        </div>

        <p style="font-size:14px; color:#555; line-height:1.6;">
            This link will expire in <strong>30 minutes</strong>.
        </p>

        <p style="font-size:14px; color:#555; line-height:1.6;">
            If the button does not work, copy and paste this link into your browser:
        </p>

        <p style="font-size:13px; color:#1f4e79; word-break:break-all;">
            {reset_link}
        </p>

        <hr style="border:none; border-top:1px solid #ddd; margin:25px 0;">

        <p style="font-size:13px; color:#777; line-height:1.6;">
            If you did not request a password reset, please ignore this email.
        </p>

        <p style="font-size:13px; color:#777;">
            Regards,<br>
            AI Job Recommendation System
        </p>
    </div>
</body>
</html>
"""

            mail.send(msg)
            flash("Password reset email sent successfully.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            print("MAIL ERROR:", repr(e))
            flash(f"Could not send reset email: {e}", "error")
            return render_template("forgot_password.html")

    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    email = verify_reset_token(token)

    if not email:
        flash("This password reset link is invalid or has expired. Please request a new one.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form["new_password"]
        confirm_password = request.form["confirm_password"]

        if new_password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("reset_password.html")

        password_hash = generate_password_hash(new_password)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE accounts
            SET password_hash = ?
            WHERE email = ?
        """, (password_hash, email))
        conn.commit()
        conn.close()

        flash("Password reset successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html")


@app.route("/")
def home():
    if not is_logged_in():
        return redirect(url_for("login"))

    return render_template("index.html", username=session.get("username"))


@app.route("/predict", methods=["POST"])
def predict():
    if not is_logged_in():
        return redirect(url_for("login"))

    experience = request.form.get("experience", "").strip()
    manual_skills = request.form.get("skills", "").strip()
    resume_file = request.files.get("resume")

    manual_list = [s.strip().lower() for s in manual_skills.split(",") if s.strip()]
    resume_list = []

    if resume_file and resume_file.filename:
        resume_text = extract_text_from_resume(resume_file)
        resume_list = extract_skills_from_resume_advanced(resume_text)

    combined_skills = sorted(set(manual_list + resume_list))
    user_skills = ", ".join(combined_skills)

    if not user_skills:
        return render_template(
            "index.html",
            username=session.get("username"),
            results=[],
            error="Please enter skills or upload a resume."
        )

    user_vector = tfidf.transform([user_skills])
    similarity = cosine_similarity(user_vector, tfidf_matrix)[0]

    sorted_indices = np.argsort(similarity)[::-1][:100]

    recommended_jobs = df.iloc[sorted_indices].copy()
    recommended_jobs["match_score"] = similarity[sorted_indices] * 100
    recommended_jobs = recommended_jobs.drop_duplicates(subset=["job_title"])
    recommended_jobs = recommended_jobs.head(5)

    final_jobs = []

    for idx, row in recommended_jobs.iterrows():
        matched_skills, missing_skills = extract_skill_sets(user_skills, row["skills_required"])

        final_jobs.append({
            "df_index": int(idx),
            "job_title": row["job_title"],
            "skills_required": row["skills_required"],
            "description": row["description"],
            "score": round(float(row["match_score"]), 2),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO users (account_id, name, email, skills, experience)
        VALUES (?, ?, ?, ?, ?)
    """, (
        session["account_id"],
        session["username"],
        session["email"],
        user_skills,
        experience
    ))
    conn.commit()
    user_id = cursor.lastrowid

    for job in final_jobs:
        row = df.loc[job["df_index"]]

        cursor.execute("""
            SELECT id FROM jobs
            WHERE job_title = ? AND description = ?
            LIMIT 1
        """, (
            row["job_title"],
            row["description"]
        ))
        result = cursor.fetchone()

        if result:
            job_id = result["id"]
        else:
            cursor.execute("""
                INSERT INTO jobs (job_title, skills_required, description)
                VALUES (?, ?, ?)
            """, (
                row["job_title"],
                row["skills_required"],
                row["description"]
            ))
            conn.commit()
            job_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO recommendations (user_id, job_id, score)
            VALUES (?, ?, ?)
        """, (
            user_id,
            job_id,
            job["score"]
        ))

    conn.commit()
    conn.close()

    return render_template(
        "index.html",
        username=session.get("username"),
        results=final_jobs,
        extracted_skills=user_skills,
        manual_skills=", ".join(manual_list),
        resume_skills=", ".join(resume_list)
    )


@app.route("/history")
def history():
    if not is_logged_in():
        return redirect(url_for("login"))

    conn = get_db_connection()

    query = """
    SELECT 
        users.id AS session_id,
        users.name,
        users.email,
        users.skills,
        users.experience,
        datetime(users.created_at, '+5 hours', '+30 minutes') AS session_time,
        jobs.job_title,
        recommendations.score
    FROM users
    LEFT JOIN recommendations ON recommendations.user_id = users.id
    LEFT JOIN jobs ON recommendations.job_id = jobs.id
    WHERE users.account_id = ?
    ORDER BY users.id DESC, recommendations.score DESC
    """

    rows = conn.execute(query, (session["account_id"],)).fetchall()
    conn.close()

    history_dict = {}

    for row in rows:
        session_id = row["session_id"]

        if session_id not in history_dict:
            history_dict[session_id] = {
                "session_id": session_id,
                "name": row["name"],
                "email": row["email"],
                "skills": row["skills"],
                "experience": row["experience"],
                "session_time": row["session_time"],
                "jobs": []
            }

        if row["job_title"] is not None and row["score"] is not None:
            history_dict[session_id]["jobs"].append({
                "job_title": row["job_title"],
                "score": row["score"]
            })

    history_data = list(history_dict.values())

    return render_template("history.html", history=history_data)


@app.route("/api/jobs", methods=["GET"])
def get_jobs():
    conn = get_db_connection()
    jobs = conn.execute("SELECT * FROM jobs LIMIT 50").fetchall()
    conn.close()

    jobs_list = [dict(job) for job in jobs]
    return jsonify(jobs_list)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()

    experience = data.get("experience", "")
    user_skills = data.get("skills", "")

    user_vector = tfidf.transform([user_skills])
    similarity = cosine_similarity(user_vector, tfidf_matrix)[0]

    sorted_indices = np.argsort(similarity)[::-1][:100]

    recommended_jobs = df.iloc[sorted_indices].copy()
    recommended_jobs["match_score"] = similarity[sorted_indices] * 100
    recommended_jobs = recommended_jobs.drop_duplicates(subset=["job_title"])
    recommended_jobs = recommended_jobs.head(5)

    recommendations = []

    for _, row in recommended_jobs.iterrows():
        matched_skills, missing_skills = extract_skill_sets(user_skills, row["skills_required"])

        recommendations.append({
            "job_title": row["job_title"],
            "skills_required": row["skills_required"],
            "description": row["description"],
            "score": round(float(row["match_score"]), 2),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })

    return jsonify({"recommendations": recommendations})


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)