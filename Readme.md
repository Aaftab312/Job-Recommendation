# AI-Powered Job Recommendation System with Database Integration

## Project Overview
This project is an AI-powered job recommendation system that suggests suitable jobs to users based on their skills and experience. It combines machine learning, database management, and backend development to simulate a real-world job recommendation platform.

## Objective
The main objective of this project is to:
- Accept user input such as name, email, skills, and experience
- Store user details in a database
- Store job details in a database
- Use machine learning to recommend the most relevant jobs
- Store recommendation history with similarity scores

## Features
- User form for entering skills and experience
- Job recommendation using TF-IDF and cosine similarity
- SQLite database integration
- Recommendation history page
- Skill gap analysis
- Match score percentage
- Clean web interface using Flask

## Technologies Used
- Python
- Flask
- SQLite
- Pandas
- NumPy
- Scikit-learn
- HTML
- CSS

## Machine Learning Approach
The system uses Natural Language Processing techniques for recommendation:
1. Job skills and descriptions are combined into text format
2. TF-IDF vectorization converts text into numerical form
3. Cosine similarity is used to compare user input with jobs
4. Top matching jobs are recommended to the user

## Database Tables
### Users
- id
- name
- email
- skills
- experience

### Jobs
- id
- job_title
- skills
- description

### Recommendations
- id
- user_id
- job_id
- score

