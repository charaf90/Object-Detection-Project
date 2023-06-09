import os

from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T


import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torchvision import datasets, models
import torch.optim as optim
from torch.autograd import Function

from torchsummary import summary


import torchvision.transforms as transforms

import cv2
import pandas as pd
import numpy as np

from PIL import Image



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
                        transforms.RandomAffine(degrees = (0,5), translate = (0.05, 0.1), scale = (0.95, 1.05)), 
                        #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 1.0)),
                        #transforms.Grayscale(num_output_channels=3), 
                        #transforms.RandomRotation((0,90)),
                        transforms.ToTensor(),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        #transforms.Normalize(mean=[0.385, 0.356, 0.806], std=[0.229, 0.224, 0.225]),
                        transforms.Normalize(mean=[0.35, 0.35, 0.35], std=[0.2, 0.2, 0.2]),

                        ])
    return transformes


def get_transform_inference():
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
                        #transforms.RandomAffine(degrees = (0,5), translate = (0.05, 0.1), scale = (0.95, 1.05)), 
                        #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 1.0)),
                        #transforms.Grayscale(num_output_channels=3), 
                        #transforms.RandomRotation((0,90)),
                        transforms.ToTensor(),
                        #transforms.RandomVerticalFlip(p=0.5),
                        #transforms.RandomHorizontalFlip(p=0.5),
                        #transforms.Normalize(mean=[0.385, 0.356, 0.806], std=[0.229, 0.224, 0.225]),
                        transforms.Normalize(mean=[0.35, 0.35, 0.35], std=[0.2, 0.2, 0.2]),

                        ])
    return transformes

def preprocess_image_detectron(image):
    """
    Effectue le prétraitement de l'image.

    Args:
        image (array): image à prétraiter.

    Returns:
        PIL.Image: Image prétraitée.
    """

    # Redimensionner l'image en forme carrée sans déformation
    size = max(image.shape[:2])
    resized_image = np.zeros((size, size, 3), dtype=np.uint8)  # Ajouter un troisième canal pour la couleur
    start_h = (size - image.shape[0]) // 2
    start_w = (size - image.shape[1]) // 2
    resized_image[start_h:start_h+image.shape[0], start_w:start_w+image.shape[1], :] = image  # Copier tous les canaux

    # Redimensionner l'image à la taille cible
    target_size = (64, 64)  # Taille cible pour l'image prétraitée
    resized_image = cv2.resize(resized_image, target_size, interpolation=cv2.INTER_AREA)

    # Convertir l'image en objet PIL Image et convertir l'espace de couleur en RGB
    preprocessed_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

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

        #Diminution du lr après 30 epochs
        if epoch == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0005

        print(f"Epoch {epoch+1}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy * 100:.2f}%")

    # Création d'un DataFrame contenant les résultats d'entraînement
    results = pd.DataFrame(list(zip(loss_train_list, loss_val_list, accuracy_train_list, accuracy_list)),
                           columns=['train_Loss', 'val_Loss', 'train_Accuracy', 'val_Accuracy'])

    return results

def prediction(im, classification_model, detection_model, transformes, device, confiance = 0.43):
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
    outputs_pred = outputs['instances'][outputs['instances'].scores > confiance]
    outputs_final = outputs_pred[outputs_pred.pred_classes < 7]

    rois = outputs_final.pred_boxes.tensor

    pred_classes_final = []
    for roi in rois:
        # Extraction de la ROI de l'image
        x1, y1, x2, y2 = roi.tolist()
        roi = im[int(y1):int(y2), int(x1):int(x2)]

        roi = preprocess_image_detectron(roi)


        #roi_image = Image.fromarray(roi)

        # Application des transformations à la ROI en utilisant torchvision.transforms
        roi_image_transformed = transformes(roi)  # Attention à ce que les transformations correspondent à ceux utiliser pour l'entraînement

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


##### GRAD CAM DETECTRON2 ######


class GradCAM():
    """
    Classe pour implémenter la fonction GradCam avec les hooks Pytorch nécessaires.

    Attributs
    ----------
    model : Modèle GeneralizedRCNN de detectron2
        Un modèle utilisant l'API detectron2 pour l'inférence
    layer_name : str
        nom de la couche convolutionnelle pour effectuer GradCAM
    """

    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradient = None
        self.model.eval()
        self.activations_grads = []
        self._register_hook()

    def _get_activations_hook(self, module, input, output):
        self.activations = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.model.named_modules():
            if name == self.target_layer_name:
                self.activations_grads.append(module.register_forward_hook(self._get_activations_hook))
                self.activations_grads.append(module.register_backward_hook(self._get_grads_hook))
                return True
        print(f"Couche {self.target_layer_name} non trouvée dans le Modèle !")

    def _release_activations_grads(self):
      for handle in self.activations_grads:
            handle.remove()
    
    def _postprocess_cam(self, raw_cam, img_width, img_height):
        cam_orig = np.sum(raw_cam, axis=0)  # [H,W]
        cam_orig = np.maximum(cam_orig, 0)  # ReLU
        cam_orig -= np.min(cam_orig)
        cam_orig /= np.max(cam_orig)
        cam = cv2.resize(cam_orig, (img_width, img_height))
        return cam, cam_orig

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._release_activations_grads()

    def __call__(self, inputs, target_category):
        """
        Appelle l'instance GradCAM++

        Paramètres
        ----------
        inputs : dict
            L'entrée dans le format standard d'entrée du modèle detectron2
            https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format

        target_category : int, optional
            L'indice de la catégorie cible. Si `None` la classe avec le score le plus élevé sera sélectionnée

        Retourne
        -------
        cam : np.array()
          Carte d'activation de classe pondérée par le gradient
        output : list
          liste d'objets Instance représentant la sortie du modèle detectron2
        """
        self.model.zero_grad()
        output = self.model.forward([inputs])

        if target_category == None:
          target_category =  np.argmax(output[0]['instances'].scores.cpu().data.numpy(), axis=-1)

        score = output[0]['instances'].scores[target_category]
        score.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        activations = self.activations[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam, cam_orig = self._postprocess_cam(cam, inputs["width"], inputs["height"])

        return cam, cam_orig, output

class GradCamPlusPlus(GradCAM):
    """
    Sous-classe pour implémenter la fonction GradCam++ avec ses hooks PyTorch nécessaires.
    ...

    Attributs
    ----------
    model : Modèle GeneralizedRCNN de detectron2
        Un modèle utilisant l'API detectron2 pour l'inférence
    target_layer_name : str
        nom de la couche convolutionnelle pour effectuer GradCAM++ 

    """
    def __init__(self, model, target_layer_name):
        super(GradCamPlusPlus, self).__init__(model, target_layer_name)

    def __call__(self, inputs, target_category):
        """
        Appelle l'instance GradCAM++

        Paramètres
        ----------
        inputs : dict
            L'entrée dans le format standard d'entrée du modèle detectron2
            https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format

        target_category : int, optional
            L'indice de la catégorie cible. Si `None` la classe avec le score le plus élevé sera sélectionnée

        Retourne
        -------
        cam : np.array()
          Carte d'activation de classe pondérée par le gradient
        output : list
          liste d'objets Instance représentant la sortie du modèle detectron2
        """
        self.model.zero_grad()
        output = self.model.forward([inputs])

        if target_category == None:
          target_category =  np.argmax(output[0]['instances'].scores.cpu().data.numpy(), axis=-1)

        score = output[0]['instances'].scores[target_category]
        score.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        activations = self.activations[0].cpu().data.numpy()  # [C,H,W]

        # de https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py
        grads_power_2 = gradient**2
        grads_power_3 = grads_power_2 * gradient
        # Equation 19 dans https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, None, None] * grads_power_3 + eps)
        # Maintenant, ramenez le ReLU de l'eq.7 dans le document,
        # Et mettez à zéro les aij où les activations sont 0
        aij = np.where(gradient != 0, aij, 0)

        weights = np.maximum(gradient, 0) * aij
        weight = np.sum(weights, axis=(1, 2))

        cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam, cam_orig = self._postprocess_cam(cam, inputs["width"], inputs["height"])

        return cam, cam_orig, output
    
class Detectron2GradCAM():
  """
      Attributs
    ----------
    cfg : ConfNd
        configuration du modèle detectron2
    model_file : str
        chemin du fichier de modèle detectron2
    """
  def __init__(self, cfg):
      self.cfg = cfg

  def _get_input_dict(self, original_image):
      # Obtient un dictionnaire d'entrée à partir d'une image originale
      height, width = original_image.shape[:2]
      # Génère la transformation de l'image
      transform_gen = T.ResizeShortestEdge(
          [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
      )
      image = transform_gen.get_transform(original_image).apply_image(original_image)
      image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)
      inputs = {"image": image, "height": height, "width": width}
      return inputs

  def get_cam(self, img, target_instance, layer_name, grad_cam_type="GradCAM"):
      """
      Appelle l'instance GradCAM++

      Paramètres
      ----------
      img : str
          Chemin vers l'image d'inférence
      target_instance : int
          L'index de l'instance cible
      layer_name : str
          Couche de convolution pour effectuer GradCAM sur
      grad_cam_type : str
          GradCAM ou GradCAM++ (pour plusieurs instances du même objet, GradCAM++ peut être préférable)

      Retourne
      -------
      image_dict : dict
        {"image" : <image>, "cam" : <cam>, "output" : <output>, "label" : <label>}
        <image> image d'entrée originale
        <cam> carte d'activation de classe redimensionnée à la forme de l'image originale
        <output> objet instances généré par le modèle
        <label> étiquette de l'instance
      cam_orig : numpy.ndarray
        cam brut non traité
      """
      model = build_model(self.cfg)
      checkpointer = DetectionCheckpointer(model)
      checkpointer.load(self.cfg.MODEL.WEIGHTS)

      image = read_image(img, format="BGR")
      input_image_dict = self._get_input_dict(image)

      # Choisissez le type de Grad CAM à utiliser
      if grad_cam_type == "GradCAM":
        grad_cam = GradCAM(model, layer_name)

      elif grad_cam_type == "GradCAM++":
        grad_cam = GradCamPlusPlus(model, layer_name)
      
      else:
        raise ValueError('Type de Grad CAM non spécifié')

      with grad_cam as cam:
        cam, cam_orig, output = cam(input_image_dict, target_category=target_instance)

      image_dict = {}
      image_dict["image"] = image
      image_dict["cam"] = cam
      image_dict["output"] = output

      return image_dict, cam_orig



##### GRAD CAM DETECTION ######


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        self.gradients = []
        outputs = []
        for name, module in self.model._modules.items():
            if name == 'fc':
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x



class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targetted layers.
    3. Gradients from intermediate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        return target_activations, output


##### GRAD CAM CLASSIFICATION ######


class GradCam_resnet50:
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img):

        features, output = self.extractor(input_img)


        target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

