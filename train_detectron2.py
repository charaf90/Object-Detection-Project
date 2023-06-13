import argparse

import util



if __name__ == "__main__":
    # Crée un analyseur d'arguments
    parser = argparse.ArgumentParser()
    # Ajoute les arguments de ligne de commande
    parser.add_argument('--class-list', default='./class.names')  # Chemin vers le fichier de liste des classes
    parser.add_argument('--data-dir', default='./data')  # Répertoire contenant les données
    parser.add_argument('--output-dir', default='./output')  # Répertoire de sortie pour enregistrer le modèle entraîné
    parser.add_argument('--device', default='cuda')  # Périphérique à utiliser pour l'entraînement (par défaut : CPU)
    parser.add_argument('--learning-rate', default=0.00025)  # Taux d'apprentissage pour l'optimiseur
    parser.add_argument('--batch-size', default=4)  # Taille du lot d'entraînement
    parser.add_argument('--iterations', default=10000)  # Nombre maximal d'itérations d'entraînement
    parser.add_argument('--checkpoint-period', default=500)  # Nombre d'itérations entre chaque sauvegarde de modèle
    parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml')  # Modèle à utiliser

    '''
    Les modèles disponibles sont accessible via cette page : https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
    '''

    # Analyse les arguments de ligne de commande
    args = parser.parse_args()

    # Appelle la fonction d'entraînement avec les arguments fournis
    util.train(args.output_dir,
               args.data_dir,
               args.class_list,
               device=args.device,
               learning_rate=float(args.learning_rate),
               batch_size=int(args.batch_size),
               iterations=int(args.iterations),
               checkpoint_period=int(args.checkpoint_period),
               model=args.model)
