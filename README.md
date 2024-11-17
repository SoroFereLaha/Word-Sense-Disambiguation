# Word Sense Disambiguation (WSD)

La d?sambigu?sation des sens des mots (WSD) consiste ? associer des mots en contexte ? leur entr?e la plus appropri?e dans un inventaire de sens pr?d?fini. L'inventaire de sens de mots pour l'anglais dans la WSD est WordNet. Par exemple, si l'on consid?re le mot ? mouse ? et la phrase suivante :

? Une souris est un objet que l'on tient dans la main et qui comporte un ou plusieurs boutons ?.

nous attribuerions ? ? souris ? le sens d'appareil ?lectronique

## ?tude Comparative et Progressive

Un projet ?volutif d'exploration et d'am?lioration des approches de d?sambigu?sation du sens des mots, combinant des m?thodes classiques, d'apprentissage automatique et de deep learning.

## ?tat Actuel
Le projet impl?mente actuellement trois approches :
- **Algorithme de Lesk** : Approche classique bas?e sur les d?finitions
- **K-Nearest Neighbors (KNN)** : M?thode d'apprentissage automatique
- **BiLSTM** : R?seau de neurones r?current bidirectionnel

## Objectifs Futurs
- Am?lioration des mod?les existants
- Int?gration de nouvelles architectures (Transformers, BERT, etc.)
- Exploration d'approches hybrides
- Optimisation des performances
- Extension ? d'autres langues

## Technologies Actuelles
- Python
- NLTK & WordNet
- Scikit-learn
- PyTorch
- Pandas & Matplotlib


## Installation

1. Cloner le repository
\\\ash
git clone https://github.com/SoroFereLaha/Word-Sense-Disambiguation.git
\\\

2. Cr?er un environnement virtuel
\\\ash
python -m venv env
source env/bin/activate  # Sur Windows: env\Scripts\activate
\\\

3. Installer les d?pendances
\\\ash
pip install -r requirements.txt
\\\

## Structure du Projet

- \src/\: Code source du projet
  - \interface/\: Interface utilisateur
  - \projet/\: Core du projet
  - \	emplates/\: Templates HTML
  - \static/\: Fichiers statiques
- \models/\: Mod?les entra?n?s
- \data/\: Donn?es d'entra?nement et de test
