import requests

# Pegunta 1

url = 'https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv'

response = requests.get(url)

if response.status_code == 200:
    with open('heart_failure_clinical_records_dataset.csv', 'r') as file:
        content = file.read()
        print(content)

#Pregunta 2

def heart_failure(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open('heart_failure_clinical_records_dataset.csv', 'r') as file:
            content = file.read()
        print(content)
   
heart_failure(url) 
