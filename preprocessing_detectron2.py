import os
import random
import xml.etree.ElementTree as ET
import shutil

# Chemins d'accès aux dossiers d'entrée et de sortie
data_folder = 'images' #Dossier contenant les images
labels_folder = 'labels' #Dossier contenant les labels
output_folder = 'data' #Dossier de sortie à ne pas modifier !

# Chemins d'accès aux dossiers d'entraînement et de validation
train_folder = os.path.join(output_folder, 'train')
val_folder = os.path.join(output_folder, 'val')

# Création des dossiers de sortie
os.makedirs(train_folder, exist_ok=True)
os.makedirs(os.path.join(train_folder, 'imgs'), exist_ok=True)
os.makedirs(os.path.join(train_folder, 'anns'), exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(os.path.join(val_folder, 'imgs'), exist_ok=True)
os.makedirs(os.path.join(val_folder, 'anns'), exist_ok=True)

# Liste des fichiers dans le dossier data_final
image_files = [file for file in os.listdir(data_folder) if file.lower().endswith('.jpg')]

# Séparation des ensembles d'entraînement et de validation
random.shuffle(image_files)
train_files = image_files[:int(len(image_files) * 0.8)]
val_files = image_files[int(len(image_files) * 0.8):]

# Conversion des annotations XML en annotations YOLO
for file in image_files:
    # Chemin d'accès à l'image d'origine
    img_path = os.path.join(data_folder, file)
    # Chemin d'accès à l'annotation XML correspondante
    label_file = os.path.splitext(file)[0] + '.xml'
    label_path = os.path.join(labels_folder, label_file)

    # Chemin d'accès au fichier de sortie pour les annotations YOLO
    output_file = os.path.splitext(file)[0] + '.txt'
    if file in train_files:
        output_path = os.path.join(train_folder, 'anns', output_file)
    else:
        output_path = os.path.join(val_folder, 'anns', output_file)

    # Analyse de l'annotation XML
    root = ET.parse(label_path).getroot()
    objects = root.findall('object')

    with open(output_path, 'w') as f:
        for obj in objects:
            xmin = int(obj.find('bndbox/xmin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymin = int(obj.find('bndbox/ymin').text)
            ymax = int(obj.find('bndbox/ymax').text)

            # Conversion en coordonnées YOLO
            img_width, img_height = root.find('size/width').text, root.find('size/height').text
            x_center = (xmin + xmax) / (2 * int(img_width))
            y_center = (ymin + ymax) / (2 * int(img_height))
            width = (xmax - xmin) / int(img_width)
            height = (ymax - ymin) / int(img_height)

            line = f'0 {x_center} {y_center} {width} {height}\n'
            f.write(line)

    # Copie de l'image vers le dossier correspondant
    if file in train_files:
        dest_folder = os.path.join(train_folder, 'imgs')
    else:
        dest_folder = os.path.join(val_folder, 'imgs')

    dest_path = os.path.join(dest_folder, file)
    shutil.copy(img_path, dest_path)

print("Conversion et division du jeu de données terminées.")
