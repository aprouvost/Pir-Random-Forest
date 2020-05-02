# First trial of creating a decision tree using python

# Classifies salaries if more than 100k salary
# based on the job, and the degreethe person has, and the company

import pandas as pd
import sklearn as sklearn
from sklearn.preprocessing import LabelEncoder
print(" ======================================================== \n")

df = pd.read_csv("salaries.csv")
# df.head()

inputs = df.drop('salary_more_than_100k', axis='columns')
target = df['salary_more_than_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
# inputs

inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')
# inputs_n

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)  # affihche les configurations de la classification
print("Score de classification de : ", model.score(inputs_n, target))  # affiche le score de la classification
print("Prédiction le type computer programming chez google d'un bachelor :", model.predict([[2, 1, 0]]))
