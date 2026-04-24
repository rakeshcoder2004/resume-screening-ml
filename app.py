import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pdfminer.high_level import extract_text

# ------------------------
# TITLE
# ------------------------

st.title("AI Resume Screening System")

st.write("Upload your resume and find best job role using Machine Learning")

# ------------------------
# LOAD DATASET
# ------------------------

df = pd.read_csv("resume_dataset.csv")

# ------------------------
# CLEAN TEXT
# ------------------------

def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

df["Resume"] = df["Resume"].apply(clean_text)

# ------------------------
# TRAIN MODEL
# ------------------------

tfidf = TfidfVectorizer(stop_words="english")

X = tfidf.fit_transform(df["Resume"])
y = df["Category"]

model = LogisticRegression(max_iter=200)

model.fit(X, y)

# ------------------------
# SKILL DATABASE
# ------------------------

skills_db = {

"programming":[
"python","java","c","c++","javascript","sql"
],

"ml":[
"machine learning","deep learning","nlp","tensorflow","pandas","numpy","sklearn"
],

"web":[
"html","css","react","node","bootstrap"
],

"cloud":[
"aws","docker","kubernetes"
],

"iot":[
"iot","arduino","raspberry pi","embedded","microcontroller","sensors"
]

}

# ------------------------
# FILE UPLOAD
# ------------------------

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:

    # save file
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # extract text
    resume_text = extract_text("temp_resume.pdf")

    cleaned_resume = clean_text(resume_text)

    # convert text to numbers
    vector = tfidf.transform([cleaned_resume])

    # prediction
    prediction = model.predict(vector)[0]

    # probability
    probs = model.predict_proba(vector)[0]

    top3_index = np.argsort(probs)[-3:][::-1]

    # ------------------------
    # OUTPUT
    # ------------------------

    st.subheader("Predicted Job Role")

    st.success(prediction)

    # ------------------------
    # TOP 3 JOBS
    # ------------------------

    st.subheader("Top 3 Suitable Jobs")

    top_jobs = []
    top_scores = []

    for i in top3_index:

        job = y.unique()[i]
        score = round(probs[i]*100,2)

        top_jobs.append(job)
        top_scores.append(score)

        st.write(job," - ",score,"%")

    # ------------------------
    # GRAPH (STEP 7)
    # ------------------------

    st.subheader("Job Match Graph")

    plt.figure()

    plt.bar(top_jobs, top_scores)

    plt.xlabel("Job Role")
    plt.ylabel("Match %")
    plt.title("Top Job Matches")

    st.pyplot(plt)

    # ------------------------
    # SKILL DETECTION
    # ------------------------

    detected_skills = []

    for category in skills_db:

        for skill in skills_db[category]:

            if skill in cleaned_resume:

                detected_skills.append(skill)

    st.subheader("Skills Found in Resume")

    st.write(detected_skills)

    # ------------------------
    # SKILL COUNT
    # ------------------------

    st.subheader("Total Skills Found")

    st.write(len(detected_skills))

    # ------------------------
    # STRONG AREA
    # ------------------------

    ml_count = 0
    prog_count = 0
    web_count = 0
    iot_count = 0

    for skill in detected_skills:

        if skill in skills_db["ml"]:
            ml_count += 1

        if skill in skills_db["programming"]:
            prog_count += 1

        if skill in skills_db["web"]:
            web_count += 1

        if skill in skills_db["iot"]:
            iot_count += 1

    st.subheader("Strong Area")

    max_area = max(ml_count, prog_count, web_count, iot_count)

    if max_area == ml_count:
        st.write("Machine Learning")

    elif max_area == prog_count:
        st.write("Programming")

    elif max_area == web_count:
        st.write("Web Development")

    else:
        st.write("IoT")

    # ------------------------
    # RESUME SCORE
    # ------------------------

    score = min(len(detected_skills)*8,100)

    st.subheader("Resume Score")

    st.write(str(score) + " / 100")

    # ------------------------
    # MISSING SKILLS
    # ------------------------

    ds_skills = [

    "python",

    "machine learning",

    "statistics",

    "sql",

    "deep learning"

    ]

    missing = []

    for skill in ds_skills:

        if skill not in detected_skills:

            missing.append(skill)

    st.subheader("Recommended Skills to Improve")

    st.write(missing)

    # ------------------------
    # CAREER SUGGESTIONS
    # ------------------------

    st.subheader("Career Suggestions")

    if "iot" in detected_skills:
        st.write("IoT Engineer")

    if "machine learning" in detected_skills:
        st.write("ML Engineer")

    if "python" in detected_skills:
        st.write("Software Developer")

    if "sql" in detected_skills:
        st.write("Data Analyst")

    # ------------------------
    # CAREER ADVICE
    # ------------------------

    st.subheader("Career Advice")

    if score < 40:
        st.write("Add more technical skills to improve resume strength")

    elif score < 70:
        st.write("Good profile, improve advanced skills")

    else:
        st.write("Strong resume for technical roles")