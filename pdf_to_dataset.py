import os
import pandas as pd
from pdfminer.high_level import extract_text

data = []

base_path = "Resume"

for category in os.listdir(base_path):
    folder_path = os.path.join(base_path, category)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)

            text = extract_text(file_path)

            # skip empty text
            if text and len(text) > 50:
                data.append([category, text])

df = pd.DataFrame(data, columns=["Category", "Resume"])

print("Total resumes extracted:", len(df))

df.to_csv("resume_dataset.csv", index=False)

print("Dataset created successfully")