from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-pt"
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Usando dispositivo: {device}")

def translate_text(text, target_language="pt"):
    
    inputs = tokenizer(f"{text}", return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=200)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def translate_dataframe(df, target_language="pt"):
    
    df['titulo'] = df['titulo'].apply(lambda x: translate_text(x, target_language))
    return df


driver = webdriver.Chrome()
driver.get("https://venturebeat.com/category/ai/")

dicionario = {'titulo': [], 'autor': [], 'data': []}

try:
    titulo = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'h2.ArticleListing__title')))
    data = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'time.ArticleListing__time')))
    autor = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a.ArticleListing__author')))
except:
    print('Erro ao tentar encontrar os elementos')


for t in titulo:
    dicionario['titulo'].append(t.text)

for a in autor:
    dicionario['autor'].append(a.text)

for d in data:
    dicionario['data'].append(d.text)


min_len = min(len(dicionario['titulo']), len(dicionario['autor']), len(dicionario['data']))
dicionario['titulo'] = dicionario['titulo'][:min_len]
dicionario['autor'] = dicionario['autor'][:min_len]
dicionario['data'] = dicionario['data'][:min_len]


df = pd.DataFrame(dicionario)


df.to_csv('venturebeat.csv', index=False)


df_traduzido = translate_dataframe(df, target_language="pt")


df_traduzido.to_csv('venturebeat_traduzido.csv', index=False)

print("Tradução concluída e salva em 'venturebeat_traduzido.csv'.")


driver.quit()
