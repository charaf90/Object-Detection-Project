# Projet de détection et de classification de véhicules à partir d'images aériennes 

## Description

Ce projet consiste à mettre en place un système de détection et de classification de véhicules à partir d'images aériennes. Il utilise deux modèles imbriqués, à savoir Detectron2 pour la détection et ResNet50 pour la classification. Le dataset utilisé dans ce projet est le VAID (Vehicle Aerial Imagery Dataset), qui a été préalablement nettoyé et préparé.

## Installation

Pour installer les dépendances nécessaires, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

## Étapes du projet

1. **Téléchargement du Dataset**: Commencez par télécharger le dataset nettoyé via ce lien : [Dataset](https://drive.google.com/drive/folders/1-I3qeZNdOT295CjSM2-6UFteI0SjL_pW?usp=drive_link). Une fois téléchargé, placez-le dans un dossier de travail approprié.

2. **Prétraitement des données**: Deux scripts sont fournis pour préparer les données pour chacun des modèles:

    - `preprocessing_classification.py`: Ce script extrait les véhicules du dataset ainsi que leur classe respective pour l'entraînement du modèle de classification. Il va créer un dossier `bbox_imgs` et un fichier `labels.csv`.

    - `preprocessing_detectron2.py`: Ce script transforme les annotations des véhicules en un type unique et les rend compatibles avec Detectron2. Les données seront stockées dans un dossier `data`, avec des sous-dossiers `train` et `val` correspondants aux données d'entraînement et de test qui seront splittées aléatoirement avec un ratio de 0.8. Dans chacun de ces dossiers, deux dossiers seront créés, `anns` pour les annotations et `imgs` pour les images.

3. **Entraînement des modèles**: Les deux modèles sont entraînés dans cette étape.

    - Le modèle de classification est entraîné grâce à un notebook Jupyter `resnet50_classification.ipynb`. Ce notebook contient également le code pour l'évaluation des performances du modèle et la sauvegarde des poids du modèle.

    - Le modèle de détection est entraîné grâce au script `train_detectron2.py`. Les performances et les poids du modèle seront enregistrés dans le dossier `output`.

4. **Test et visualisation du modèle**: Un notebook Jupyter `detection_classification_test.ipynb` a été créé pour tester le modèle et visualiser les détections. Ce notebook utilise le package FiftyOne pour évaluer le modèle sur différentes métriques.

5. **Fichier utilitaire**: Le fichier `util.py` contient toutes les classes et fonctions qui sont utilisées tout au long du projet.


## Structure du Projet

Voici la structure final du projet:

Projet
```bash
├── README.md
├── preprocessing_classification.py           
├── preprocessing_detectron2.py
├── resnet50_classification.ipynb
├── train_detectron2.py
├── detection_classification_test.ipynb
├── util.py
├── model_final.pth                           #Modele final de détection
├── vaid_classifier.pth                       #Modele final de classification
├── data
│ ├── train
│ │ ├── anns
│ │ └── imgs
│ └── val
│ ├── anns
│ └── imgs
├── bbox_imgs                                 #Véhicules extraits
├── labels.csv                                #Classe de chaque véhicule
└── output
  └── metrics.json                            #Métriques d'évaluation du modèle de detection

```
## Utilisations du modèle

Voici quelques exemples de ce que le modèle est capable de faire :

![Exemple 1](Exemples/exemple1.png)
*Exemple de détection et de classification sur une image aérienne*

![Exemple 2](Exemples/exemple2.png)
*Un autre exemple de détection et de classification sur une image aérienne*

### Utilisations du modèle avec fiftyone

![Exemple 2](Exemples/gifexemple.gif)

*Exemple avec utilisation de fiftyone*
