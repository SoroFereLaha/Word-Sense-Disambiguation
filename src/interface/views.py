from django.shortcuts import render
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

# Configuration initiale
tf.keras.backend.clear_session()

# Chargement des données
df = pd.read_csv('semcor_clean.csv')
sentences = df['sentence'].tolist()
target_words = df['target_word'].tolist()
sense_keys = df['sense'].tolist()

# Préparation des encodeurs et modèles
label_encoder = LabelEncoder()
label_encoder.fit_transform(sense_keys)

knnModel = KNeighborsClassifier(n_neighbors=3)
with open('modele_knn.pkl', 'rb') as file:
    knnModel.__dict__.update(pickle.load(file).__dict__)

def preprocess(sentence):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    return tokens

# Préparation du Word2Vec
processed_sentences = [preprocess(sentence) for sentence in sentences]
model = Word2Vec(processed_sentences, min_count=1)

def sentence_to_vec(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(model.vector_size)

# Préparation du tokenizer et du modèle LSTM
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

try:
    modelLSTM = load_model("BiLSTM_with_target_word.h5", compile=False, custom_objects={'time_major': None})
except ValueError as e:
    print(f"Erreur lors du chargement du modèle : {e}")

def prepare_input(sentence, target_word, max_length):
    try:
        sequence = tokenizer.texts_to_sequences([sentence])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        target_word_sequence = tokenizer.texts_to_sequences([target_word])[0]

        if not target_word_sequence:
            target_word_id = 0
        else:
            target_word_id = target_word_sequence[0]

        return sequence, target_word_id
    except Exception as e:
        print(f"Erreur dans prepare_input : {e}")
        return None, None

# Vues
def home(request):
    return render(request, 'pages/home.html')

def lesk_method(request):
    if request.method == 'POST':
        context_words = nltk.word_tokenize(request.POST.get('phrase'))
        sense = lesk(context_words, request.POST.get('word'))

        if sense is not None:
            sense_key = 'Sense Key : ' + sense.name()
            definition = 'Definition : ' + sense.definition()
        else:
            sense_key = "Aucun sens trouvé pour le mot donné."
            definition = "Définition non disponible."

        context = {
            'phrase': 'Texte : ' + request.POST.get('phrase'),
            'word': 'Mot cible : ' + request.POST.get('word'),
            'sense_key': sense_key,
            'definition': definition
        }
        return render(request, 'pages/lesk.html', context)
    
    return render(request, 'pages/lesk.html', {'phrase': ' ', 'word': ' '})


def knn(request):
    if request.method == 'POST':
        query_sentence = request.POST.get('phrase')
        query_target_word = request.POST.get('word')

        query_sentence_vector = sentence_to_vec(query_sentence, model)
        query_target_word_vector = model.wv[query_target_word] if query_target_word in model.wv else np.zeros(model.vector_size)
        vector_sentence_word = np.concatenate((query_sentence_vector, query_target_word_vector)).reshape(1, -1)
        predicted_label_num = knnModel.predict(vector_sentence_word)
        predicted_sense_key = label_encoder.inverse_transform(predicted_label_num)[0]
        synset = wordnet.lemma_from_key(predicted_sense_key).synset()
        context = {
            'phrase': 'Texte : '+request.POST.get('phrase'),
            'word': 'Mot cible : '+request.POST.get('word'),
            'sense_key': 'Sense Key : '+predicted_sense_key,
            'definition': 'Definition : '+synset.definition()
        }
        return render(request, 'pages/knn.html', context)
    return render(request, 'pages/knn.html', {'phrase': ' ', 'word': ' '})

def bilstm(request):
    if request.method == 'POST':
        try:
            new_sentence = request.POST.get('phrase')
            target_word = request.POST.get('word')

            sequence, target_word_id = prepare_input(new_sentence, target_word, 86)
            if sequence is None or target_word_id is None:
                raise ValueError("Erreur dans la préparation des données d'entrée")

            prediction = modelLSTM.predict([sequence, np.array([target_word_id])])
            predicted_sense = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            synset = wordnet.lemma_from_key(predicted_sense).synset()
            context = {
                'phrase': 'Texte : ' + new_sentence,
                'word': 'Mot cible : ' + target_word,
                'sense_key': 'Sense Key : ' + predicted_sense,
                'definition': 'Definition : ' + synset.definition()
            }
        except Exception as e:
            context = {
                'error': f"Une erreur s'est produite : {str(e)}",
                'phrase': 'Texte : ' + new_sentence,
                'word': 'Mot cible : ' + target_word
            }
        return render(request, 'pages/bilstm.html', context)
    return render(request, 'pages/bilstm.html', {'phrase': ' ', 'word': ' '})

from django.http import HttpResponse
def index(request):
    return HttpResponse("le test fonctionne !")
