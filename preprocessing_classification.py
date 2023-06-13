import os
import cv2
import pandas as pd
import xml.etree.ElementTree as ET

def extract_image_regions(df, output_dir='bbox_imgs', csv_file='labels.csv'):
    # Créer le dossier pour enregistrer les images de boîtes englobantes
    os.makedirs(output_dir, exist_ok=True)

    # Créer un DataFrame pour stocker les noms d'image et les étiquettes
    filenames = []
    labels = []
    k = 1
    # Parcourir chaque ligne du DataFrame
    for i, row in df.iterrows():
        # Extraire le nom de fichier, les coordonnées de la boîte englobante et les étiquettes de classe
        filename = row['filename']
        bndboxes = row['bndbox']
        obj_labels = row['object']
        image = cv2.imread('images/' + filename)

        # Parcourir chaque boîte englobante et son étiquette de classe correspondante
        for j, (bbox, obj_label) in enumerate(zip(bndboxes, obj_labels)):
            # Extraire la région d'intérêt (ROI) en utilisant les coordonnées de la boîte englobante
            roi = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            roi_height, roi_width = roi.shape[:2]

            # Enregistrer la ROI comme une image numérotée dans le dossier des images de boîtes englobantes
            roi_filename = f'bndbox_{k}.jpg'
            roi_filepath = os.path.join(output_dir, roi_filename)
            # Il y a un problème d'annotation sur une image, je vérifie si l'image à une taille avant de l'enregistrer sinon elle passe son tour
            if roi_height > 0 and roi_width > 0:
                cv2.imwrite(roi_filepath, roi)
            # Ajouter le nom d'image et l'étiquette au DataFrame des étiquettes
                filenames.append(roi_filename)
                labels.append(obj_label)
                k+=1
    # Enregistrer le DataFrame des étiquettes dans un fichier CSV
    labels_df = pd.DataFrame({'filename': filenames, 'label': labels})
    labels_df.to_csv(csv_file, index=False)


def create_annotation_dataframe(path):
    # Initialisation d'un DataFrame vide
    annot = pd.DataFrame()
    
    # Pour chaque fichier dans le répertoire spécifié par le chemin `path`
    for file in os.listdir(str(path)):
        # On ouvre le fichier XML avec la bibliothèque ElementTree
        tree = ET.parse(path+file)
        root = tree.getroot()

        # On crée un dictionnaire avec les informations du fichier XML
        data = {
            'filename': root.find('filename').text,
            'width': int(root.find('size/width').text),
            'height': int(root.find('size/height').text),
            'depth': int(root.find('size/depth').text),
            'object': [obj.find('name').text for obj in root.findall('object')],
            'bndbox': [
                [int(bbox.find('xmin').text), int(bbox.find('ymin').text), 
                 int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
                for bbox in root.findall('object/bndbox')
            ]
        }

        # On ajoute les informations de chaque fichier dans le DataFrame
        annot = pd.concat([annot, pd.DataFrame.from_dict(data, orient='index').T], axis = 0)
    
    # On convertit certaines colonnes en type entier ou chaîne de caractères
    annot['filename'] = annot['filename'].astype('string')
    annot['width'] = annot['width'].astype(int)
    annot['height'] = annot['height'].astype(int)
    
    # On ajoute une colonne pour le nombre de détections dans chaque fichier
    annot['nb_detection'] = annot['object'].apply(lambda x: len(x))
    
    # On ajoute des colonnes pour chaque classe de véhicule, en comptant le nombre de détections pour chaque classe
    annot['sedan'] = annot['object'].apply(lambda x: x.count('1'))
    annot['minibus'] = annot['object'].apply(lambda x: x.count('2'))
    annot['truck'] = annot['object'].apply(lambda x: x.count('3'))
    annot['pickup'] = annot['object'].apply(lambda x: x.count('4'))
    annot['bus'] = annot['object'].apply(lambda x: x.count('5'))
    annot['ement truck'] = annot['object'].apply(lambda x: x.count('6'))
    annot['trailer'] = annot['object'].apply(lambda x: x.count('7'))

    # On retourne le DataFrame final
    return annot

# On créer un dataframe (qui m'a servit à visualiser les données au passage)
vaid_final = create_annotation_dataframe('labels/')

# On extrait les bndbox en les enregistrant dans un dossier nommé 'bbox_imgs' et un csv 'labels.csv'
extract_image_regions(vaid_final, output_dir='bbox_imgs', csv_file='labels.csv')
