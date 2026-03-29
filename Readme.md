# 🚀 AI Job Recommendation System

An AI-powered web application that recommends jobs based on user skills using Machine Learning techniques like **TF-IDF** and **Cosine Similarity**.

---

## 📌 Features

- 🔍 Job recommendation based on user skills  
- 📊 Skill gap analysis (matched & missing skills)  
- 📄 Resume parsing (PDF/TXT support)  
- 🔐 User authentication (Signup/Login)  
- 🔑 Password reset via email  
- 📁 Recommendation history tracking  
- 🌐 REST API for job recommendations  

---

## 🧠 Machine Learning Approach

- Text preprocessing using job skills and descriptions  
- TF-IDF vectorization  
- Cosine similarity for matching user skills with job data  

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask  
- **Machine Learning:** Scikit-learn, TF-IDF  
- **Database:** SQLite  
- **Frontend:** HTML, CSS  
- **NLP:** spaCy  
- **Other Libraries:** RapidFuzz, PyPDF2  

---

## 📂 Project Structure


project/
│
├── app.py

├── model.py

├── model.pkl

├── requirements.txt

├── .env.example

├── create_db.py

├── insert__jobs.py
│
├── templates/

├── static/


---

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/Aaftab312/Job-Recommendation.git

cd Job-Recommendation


---

### 2. Install dependencies

pip install -r requirements.txt


---

### 3. Setup environment variables

Create a `.env` file in the root directory and add:


MAIL_USERNAME=your_email@gmail.com

MAIL_PASSWORD=your_app_password

SECRET_KEY=your_secret_key

SECURITY_PASSWORD_SALT=your_salt


> ⚠️ Note: `.env` file is not included in the repository for security reasons.

---

### 4. Train the model

python model.py


---

### 5. Setup database

python create_db.py
python insert__jobs.py


---

### 6. Run the application

python app.py


---

## 📧 Email Feature

The password reset feature requires a valid Gmail App Password.

- If not configured, email functionality may not work  
- Other features will still work normally  

---

## 🎥 Demo Video

👉 Add your Google Drive / YouTube link here

---

## 🔐 Security Note

Sensitive information like email credentials is stored in a `.env` file and excluded using `.
