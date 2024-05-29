import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import spacy

# Cargar los datos
df = pd.read_csv('tu_archivo.csv')
textos = df['nombre_de_tu_columna']

# Descargar recursos de NLTK si no están descargados
nltk.download('punkt')
nltk.download('stopwords')

# Cargar SpaCy para lematización en español
nlp = spacy.load('es_core_news_sm')

# Pre-procesamiento
def limpiar_texto(texto):
    # Minúsculas
    texto = texto.lower()
    # Eliminar emojis y otros caracteres no alfabéticos
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'[0-9]+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    # Tokenización
    palabras = word_tokenize(texto, language='spanish')
    # Eliminar palabras cortas y stop words
    palabras = [palabra for palabra in palabras if len(palabra) > 2]
    stop_words = set(stopwords.words('spanish'))
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    # Lematización
    texto_lematizado = " ".join([token.lemma_ for token in nlp(" ".join(palabras))])
    return texto_lematizado

# Aplicar limpieza a cada texto
df['texto_limpio'] = textos.apply(limpiar_texto)

# Guardar el resultado en un nuevo CSV
df.to_csv('textos_procesados.csv', index=False)
