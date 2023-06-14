import os

from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo
import detectron2.utils.comm as comm

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torchvision import datasets, models
import torch.optim as optim



import torchvision.transforms as transforms

import cv2
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm


def get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes):
    """
    Crée un objet de configuration Detectron2 et définissez ses attributs.

    Args:
        output_dir (str): Le chemin du répertoire de sortie où le modèle entraîné et les journaux seront enregistrés.
        learning_rate (float): Le taux d'apprentissage pour l'optimiseur.
        batch_size (int): La taille du lot utilisée lors de l'entraînement.
        iterations (int): Le nombre maximum d'itérations d'entraînement.
        checkpoint_period (int): Le nombre d'itérations entre les points de contrôle consécutifs.
        model (str): Le nom du modèle à utiliser, qui doit être l'un des modèles disponibles dans le zoo de modèles de Detectron2.
        device (str): L'appareil à utiliser pour l'entraînement, qui doit être 'cpu' ou 'cuda'.
        nmr_classes (int): Le nombre de classes dans l'ensemble de données.

    Returns:
        L'objet de configuration Detectron2.
    """
    cfg = _get_cfg()

    # Fusionne le fichier de configuration par défaut du modèle avec le fichier de configuration par défaut de Detectron2.
    cfg.merge_from_file(model_zoo.get_config_file(model))

    # Définit les ensembles de données d'entraînement et de validation et exclut l'ensemble de données de test.
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.VAL = ("val",)
    cfg.DATASETS.TEST = ()

    # Définit l'appareil à utiliser pour l'entraînement. (Je suis sur MAC, utilisé "gpu" si colab ou carte graphique)
    cfg.MODEL.DEVICE = device

    # Définit le nombre de travailleurs du chargeur de données.
    cfg.DATALOADER.NUM_WORKERS = 2

    # Définit les poids du modèle sur ceux pré-entraînés sur l'ensemble de données COCO.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # Définit la taille du lot utilisée par le solveur.
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    # Définit la période de point de contrôle.
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    # Définit le taux d'apprentissage de base.
    cfg.SOLVER.BASE_LR = learning_rate

    # Définit le nombre maximum d'itérations d'entraînement.
    cfg.SOLVER.MAX_ITER = iterations

    # Définit les étapes de l'ordonnanceur du taux d'apprentissage sur une liste vide, ce qui signifie que le taux d'apprentissage ne sera pas réduit.
    cfg.SOLVER.STEPS = []

    # Définit la taille du lot utilisée par les têtes ROI lors de l'entraînement.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Définit le nombre de classes.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes

    # Définit le répertoire de sortie.
    cfg.OUTPUT_DIR = output_dir
    

    return cfg



def get_dicts(img_dir, ann_dir):
    """
    Lit les annotations du jeu de données au format YOLO et crée une liste de dictionnaires contenant des informations pour chaque
    image.

    Args:
        img_dir (str): Répertoire contenant les images.
        ann_dir (str): Répertoire contenant les annotations.

    Returns:
        list[dict]: Une liste de dictionnaires contenant des informations pour chaque image. Chaque dictionnaire a les clés suivantes :
            - file_name : Le chemin vers le fichier image.
            - image_id : L'identifiant unique de l'image.
            - height : La hauteur de l'image en pixels.
            - width : La largeur de l'image en pixels.
            - annotations : Une liste de dictionnaires, un pour chaque objet dans l'image, contenant les clés suivantes :
                - bbox : Une liste de quatre entiers [x0, y0, w, h] représentant la boîte englobante de l'objet dans l'image,
                          où (x0, y0) est le coin supérieur gauche et (w, h) sont la largeur et la hauteur de la boîte englobante,
                          respectivement.
                - bbox_mode : Une constante de la classe `BoxMode` indiquant le format des coordonnées de la boîte englobante
                              (par exemple, `BoxMode.XYWH_ABS` pour des coordonnées absolues au format [x0, y0, w, h]).
                - category_id : L'ID entier de la classe de l'objet.
    """
    dataset_dicts = []
    for idx, file in enumerate(os.listdir(ann_dir)):
        # les annotations doivent être fournies au format YOLO

        record = {}

        filename = os.path.join(img_dir, file[:-4] + '.jpg')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        with open(os.path.join(ann_dir, file)) as r:
            lines = [l[:-1] for l in r.readlines()]

        for _, line in enumerate(lines):
            if len(line) > 2:
                label, cx, cy, w_, h_ = line.split(' ')

                obj = {
                    "bbox": [int((float(cx) - (float(w_) / 2)) * width),
                             int((float(cy) - (float(h_) / 2)) * height),
                             int(float(w_) * width),
                             int(float(h_) * height)],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": int(label),
                }

                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts



def register_datasets(root_dir, class_list_file):
    """
    Enregistre les ensembles de données d'entraînement et de validation et retourne le nombre de classes.

    Args:
        root_dir (str): Chemin vers le répertoire racine de l'ensemble de données.
        class_list_file (str): Chemin vers le fichier contenant la liste des noms de classe.

    Returns:
        int: Le nombre de classes dans l'ensemble de données.
    """
    # Lire la liste des noms de classe à partir du fichier de liste de classe.
    with open(class_list_file, 'r') as reader:
        classes_ = [l[:-1] for l in reader.readlines()]

    # Enregistrer les ensembles de données d'entraînement et de validation.
    for d in ['train', 'val']:
        DatasetCatalog.register(d, lambda d=d: get_dicts(os.path.join(root_dir, d, 'imgs'),
                                                         os.path.join(root_dir, d, 'anns')))
        # Définir les métadonnées pour l'ensemble de données.
        MetadataCatalog.get(d).set(thing_classes=classes_)

    return len(classes_)



class ValidationLoss(HookBase):
    """
    Classe qui calcule la perte de validation pendant l'entraînement.

    Attributs:
        cfg (CfgNode): Le nœud de configuration de detectron2.
        _loader (itérateur): Un itérateur sur l'ensemble de données de validation.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): Le nœud de configuration de detectron2.
        """
        super().__init__()
        self.cfg = cfg.clone()
        # Passer à l'ensemble de données de validation
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        # Construire l'itérateur du chargeur de données de validation
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        """
        Calcule la perte de validation après chaque étape d'entraînement.
        """
        # Obtenir le prochain lot de données à partir du chargeur de données de validation
        data = next(self._loader)
        with torch.no_grad():
            # Calculer la perte de validation sur le lot de données actuel
            loss_dict = self.trainer.model(data)

            # Vérifier les pertes invalides
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            # Réduire la perte
            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # Enregistrer la perte de validation dans le stockage de l'entraîneur
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)



def train(output_dir, data_dir, class_list_file, learning_rate, batch_size, iterations, checkpoint_period, device,
          model):
    """
    Entraîne un modèle Detectron2 sur un ensemble de données personnalisé.

    Args:
        output_dir (str): Chemin vers le répertoire où enregistrer le modèle entraîné et les fichiers de sortie.
        data_dir (str): Chemin vers le répertoire contenant l'ensemble de données.
        class_list_file (str): Chemin vers le fichier contenant la liste des noms de classe dans l'ensemble de données.
        learning_rate (float): Taux d'apprentissage pour l'optimiseur.
        batch_size (int): Taille du lot pour l'entraînement.
        iterations (int): Nombre maximal d'itérations d'entraînement.
        checkpoint_period (int): Nombre d'itérations après lesquelles sauvegarder une version du modèle.
        device (str): Appareil à utiliser pour l'entraînement (par exemple, 'cpu' ou 'cuda' ou 'mps').
        model (str): Nom de la configuration du modèle à utiliser. Doit être une clé dans le zoo de modèles Detectron2.

    Returns:
        None
    """

    # Enregistrer l'ensemble de données et obtenir le nombre de classes
    nmr_classes = register_datasets(data_dir, class_list_file)

    # Obtenir la configuration du modèle
    cfg = get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes)

    # Créer le répertoire de sortie
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Créer l'objet d'entraînement
    trainer = DefaultTrainer(cfg)

    # Créer un objet de perte de validation personnalisé
    val_loss = ValidationLoss(cfg)

    # Enregistrer l'objet de perte de validation personnalisé en tant que crochet dans l'entraîneur
    trainer.register_hooks([val_loss])

    # Inverser les positions d'évaluation et de sauvegarde afin que la perte de validation soit correctement enregistrée
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    # Reprendre l'entraînement à partir d'une version sauvegardée ou charger les poids initiaux du modèle
    trainer.resume_or_load(resume=False)

    # Entraîner le modèle
    trainer.train()



def get_transform():
    """
    Applique les transformations nécessaires pour la classification.

    Attention, cette fonction est utilisé pour la classification d'image isolée et également la classifiction des images extraites des détéctections de detectron2.
    Si vous modifiez ces transformations veuillez entrainement une nouvelle fois le modèle de classification pour que les images des détéctions de detectron2 respecte les paramètres d'entrainement.

    Args:
        None

    Returns:
        Retourne l'objet transforms
    """
    transformes = transforms.Compose([
                        transforms.Resize((64, 64)),
                        #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 1.0)),
                        #transforms.Grayscale(num_output_channels=3), 
                        #transforms.RandomRotation((0,90)),
                        transforms.ToTensor(),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        #transforms.Normalize(mean=[0.385, 0.356, 0.806], std=[0.229, 0.224, 0.225]),
                        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.2, 0.2, 0.2]),

                        ])
    return transformes

from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Effectue le prétraitement de l'image.

    Args:
        image_path (str): Chemin de l'image à prétraiter.

    Returns:
        PIL.Image: Image prétraitée.
    """
    # Charger l'image en niveaux de gris avec OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Redimensionner l'image en forme carrée sans déformation
    size = max(image.shape[:2])
    resized_image = np.zeros((size, size), dtype=np.uint8)
    start_h = (size - image.shape[0]) // 2
    start_w = (size - image.shape[1]) // 2
    resized_image[start_h:start_h+image.shape[0], start_w:start_w+image.shape[1]] = image

    # Redimensionner l'image à la taille cible
    target_size = (128, 128)  # Taille cible pour l'image prétraitée
    resized_image = cv2.resize(resized_image, target_size, interpolation=cv2.INTER_AREA)

    # Appliquer un filtre Gaussien de taille de kernel 3x3
    blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)
    # Appliquer les filtres de Sobel en x et y
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_16S, 0, 1, ksize=3)


    # Mettre en valeur les contours des objets en combinant les filtres Sobel
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    enhanced_edges = cv2.addWeighted(sobel_x, 0.8, sobel_y, 0.8, 0)

    # Convertir l'image en objet PIL Image
    preprocessed_image = Image.fromarray(enhanced_edges)

    return preprocessed_image


class VaidVehiculeDataset_test(Dataset):
    """
    Dataset personnalisé pour les données de validation des véhicules.

    Args:
        annotations_file (str): Chemin vers le fichier d'annotations.
        img_dir (str): Répertoire contenant les images.
        transformes (callable, optional): Transformations à appliquer aux images.

    Attributes:
        img_labels (DataFrame): Données d'annotations des images.
        img_dir (str): Répertoire des images.
        transform (callable): Transformations à appliquer aux images.
    """

    def __init__(self, annotations_file, img_dir, transformes=None):
        """
        Initialise le dataset de validation des véhicules.

        Args:
            annotations_file (str): Chemin vers le fichier d'annotations.
            img_dir (str): Répertoire contenant les images.
            transformes (callable, optional): Transformations à appliquer aux images.
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transformes

    def __len__(self):
        """
        Retourne la taille du dataset (nombre d'images).

        Returns:
            int: Taille du dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Récupère un élément du dataset à partir de son index.

        Args:
            idx (int): Index de l'élément à récupérer.

        Returns:
            tuple: Tuple contenant l'image et son label correspondant.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        # Prétraitement de l'image en utilisant la fonction indépendante preprocess_image
        preprocessed_image = preprocess_image(img_path)

        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            preprocessed_image = self.transform(preprocessed_image)

        return preprocessed_image, label




class VaidVehiculeDataset(Dataset):
    """
    Dataset personnalisé pour les données de validation des véhicules.

    Args:
        annotations_file (str): Chemin vers le fichier d'annotations.
        img_dir (str): Répertoire contenant les images.
        transformes (callable, optional): Transformations à appliquer aux images.

    Attributes:
        img_labels (DataFrame): Données d'annotations des images.
        img_dir (str): Répertoire des images.
        transform (callable): Transformations à appliquer aux images.
    """

    def __init__(self, annotations_file, img_dir, transformes=None):
        """
        Initialise le dataset de validation des véhicules.

        Args:
            annotations_file (str): Chemin vers le fichier d'annotations.
            img_dir (str): Répertoire contenant les images.
            transformes (callable, optional): Transformations à appliquer aux images.
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transformes

    def __len__(self):
        """
        Retourne la taille du dataset (nombre d'images).

        Returns:
            int: Taille du dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Récupère un élément du dataset à partir de son index.

        Args:
            idx (int): Index de l'élément à récupérer.

        Returns:
            tuple: Tuple contenant l'image et son label correspondant.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label



class Classifier(nn.Module):
    """
    Classe du modèle de classification.

    Args:
        output_size (int): Taille de la couche de sortie du modèle.
        pretrained (bool, optional): Indique si le modèle pré-entraîné doit être utilisé.

    Attributes:
        model (nn.Module): Modèle de base utilisé (ici, ResNet-50).
    """

    def __init__(self, output_size, pretrained=True):
        """
        Initialise le modèle de classification.

        Args:
            output_size (int): Taille de la couche de sortie du modèle.
            pretrained (bool, optional): Indique si le modèle pré-entraîné doit être utilisé.
        """
        super().__init__()
        self.model = models.resnet50(pretrained = pretrained)

        # Remplace la dernière couche linéaire du modèle par une nouvelle couche avec la taille de sortie souhaitée
        self.model.fc = nn.Linear(self.model.fc.in_features, output_size)
        #self.model.classifier[1] = nn.Linear(in_features=2560, out_features=output_size)

    def forward(self, x):
        """
        Passe avant du modèle.

        Args:
            x (torch.Tensor): Entrée du modèle.

        Returns:
            torch.Tensor: Sortie du modèle.
        """
        out = self.model(x)
        return out



    

def classification_train(model, train_dl, val_dl, num_classes=7, learning_rate=0.001, batch_size=32, num_epochs=20, device="cuda"):
    """
    Entraîne le modèle de classification.

    Args:
        model (nn.Module): Modèle de classification.
        num_classes (int): Nombre de classes dans le jeu de données.
        learning_rate (float): Taux d'apprentissage pour l'optimiseur.
        batch_size (int): Taille des lots pour l'entraînement.
        num_epochs (int): Nombre d'époques d'entraînement.
        device (str): Appareil sur lequel l'entraînement doit être effectué (par défaut, "cuda").

    Returns:
        pandas.DataFrame: Résultats d'entraînement comprenant les pertes d'entraînement et de validation, ainsi que les
                          précisions d'entraînement et de validation.
    """
    loss_train_list = []  # Liste des pertes d'entraînement
    accuracy_train_list = []  # Liste des précisions d'entraînement
    loss_val_list = []  # Liste des pertes de validation
    accuracy_list = []  # Liste des précisions de validation

    iteration_list = []  # Liste des itérations

    iterations = 0  # Compteur d'itérations

    criterion = nn.CrossEntropyLoss()  # Fonction de perte (entropie croisée)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimiseur (Adam)

    for epoch in range(num_epochs):
        model.train()  # Mode d'entraînement
        total_correct = 0  # Nombre total de prédictions correctes
        total_samples = 0  # Nombre total d'échantillons
        for images, labels in tqdm(train_dl):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels - 1)  # Calcul de la perte
            loss.backward()  # Rétropropagation du gradient
            optimizer.step()  # Mise à jour des poids

            _, predicted = torch.max(outputs, dim=1)  # Récupération des prédictions
            total_samples += labels.size(0)
            total_correct += (predicted == labels - 1).sum().item()  # Calcul du nombre de prédictions correctes

            iterations += 1

        loss_train_list.append(loss.data)
        accuracy = total_correct / total_samples
        accuracy_train_list.append(accuracy)

        # Évaluation du modèle
        model.eval()  # Mode d'évaluation
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(val_dl):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                val_loss = criterion(outputs, labels - 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels - 1).sum().item()

        loss_val_list.append(val_loss.data)
        iteration_list.append(iterations)

        accuracy = total_correct / total_samples
        accuracy_list.append(accuracy)
        iterations += 1

        print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy * 100:.2f}%")

    # Création d'un DataFrame contenant les résultats d'entraînement
    results = pd.DataFrame(list(zip(loss_train_list, loss_val_list, accuracy_train_list, accuracy_list)),
                           columns=['train_Loss', 'val_Loss', 'train_Accuracy', 'val_Accuracy'])

    return results

def prediction(im, classification_model, detection_model, transformes, device):
    """
    Effectue une prédiction à l'aide d'un modèle de détection d'objets et d'un modèle de classification.

    Args:
        im (PIL.Image.Image): Image sur laquelle effectuer la prédiction.
        classification_model (nn.Module): Modèle de classification utilisé pour classer les régions d'intérêt (ROIs).
	    detection_model (nn.Module): Modèle de detection utilisé pour detecter les régions d'intérêt (ROIs).
        transformes (torchvision.transforms):  Transformations des ROIs.
        device (str) : gpu, cuda, mps or cpu 

    Returns:
        torch.Tensor: Les sorties finales de la prédiction, contenant les régions d'intérêt (ROIs) et les classes prédites.
    """

    outputs = detection_model(im) 
    outputs_pred = outputs['instances'][outputs['instances'].scores > 0.4]
    outputs_final = outputs_pred[outputs_pred.pred_classes < 7]

    rois = outputs_final.pred_boxes.tensor

    pred_classes_final = []
    for roi in rois:
        # Extraction de la ROI de l'image
        x1, y1, x2, y2 = roi.tolist()
        roi = im[int(y1):int(y2), int(x1):int(x2)]

        roi = preprocess_image(roi)


        roi_image = Image.fromarray(roi)

        # Application des transformations à la ROI en utilisant torchvision.transforms
        roi_image_transformed = transformes(roi_image)  # Attention à ce que les transformations correspondent à ceux utiliser pour l'entraînement

        # Conversion de l'image transformée en tenseur
        roi_image_tensor = roi_image_transformed.unsqueeze(0)  # Ajout d'une dimension batch
        roi_image_tensor = roi_image_tensor.to(device)
        
        # Classification de la ROI avec le modèle de classification
        classification_output = classification_model(roi_image_tensor)
        predicted_class = torch.argmax(classification_output)
        pred_classes_final.append(predicted_class.item())

    # Remplacement de la classe prédite par la classe de classification
    outputs_final.pred_classes = torch.tensor(pred_classes_final)
    return outputs_final


def prediction_for_fiftyone(im, classification_model, detection_model, transformes, device):
    """
    Effectue une prédiction à l'aide d'un modèle de détection d'objets et d'un modèle de classification.

    Args:
        im (PIL.Image.Image): Image sur laquelle effectuer la prédiction.
        classification_model (nn.Module): Modèle de classification utilisé pour classer les régions d'intérêt (ROIs).
	    detection_model (nn.Module): Modèle de detection utilisé pour detecter les régions d'intérêt (ROIs).
        transformes (torchvision.transforms):  Transformations des ROIs.
        device (str) : gpu, cuda, mps or cpu 

    Returns:
        torch.Tensor: Les sorties finales de la prédiction, contenant les régions d'intérêt (ROIs) et les classes prédites.
    """

    outputs = detection_model(im) 
    outputs['instances'] = outputs['instances'][outputs['instances'].scores > 0.4]
    outputs['instances'] = outputs['instances'][outputs['instances'].pred_classes < 7]

    rois = outputs.pred_boxes.tensor

    pred_classes_final = []
    for roi in rois:
        # Extraction de la ROI de l'image
        x1, y1, x2, y2 = roi.tolist()
        roi = im[int(y1):int(y2), int(x1):int(x2)]
        roi_image = Image.fromarray(roi)

        # Application des transformations à la ROI en utilisant torchvision.transforms
        roi_image_transformed = transformes(roi_image)  # Attention à ce que les transformations correspondent à ceux utiliser pour l'entraînement

        # Conversion de l'image transformée en tenseur
        roi_image_tensor = roi_image_transformed.unsqueeze(0)  # Ajout d'une dimension batch
        roi_image_tensor = roi_image_tensor.to(device)
        
        # Classification de la ROI avec le modèle de classification
        classification_output = classification_model(roi_image_tensor)
        predicted_class = torch.argmax(classification_output)
        pred_classes_final.append(predicted_class.item())

    # Remplacement de la classe prédite par la classe de classification
    outputs.pred_classes = torch.tensor(pred_classes_final)
    return outputs
