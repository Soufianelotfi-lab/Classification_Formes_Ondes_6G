# Classification_Formes_Ondes_6G
## Présentation du projet
Ce projet a pour objectif de classifier différentes formes d’ondes 6G à partir de signaux I/Q, en s’appuyant sur des techniques de Deep Learning. Les signaux étudiés correspondent à différentes modulations et formes d’ondes, dont les caractéristiques peuvent être difficiles à distinguer par des méthodes classiques.

Les données sont analysées selon deux représentations complémentaires. Dans le domaine temporel, le modèle exploite directement l’évolution des échantillons I/Q afin d’apprendre les dynamiques temporelles propres à chaque forme d’onde. Dans le domaine fréquentiel, les signaux sont transformés à l’aide de la FFT afin de mettre en évidence leur contenu spectral, ce qui permet de capturer des signatures fréquentielles caractéristiques.

## Organisation du projet
Le projet est structuré autour de deux notebooks principaux, chacun correspondant à une approche de modélisation.
#### Notebook 1 – Modèles CNN + LSTM
- Ce premier notebook regroupe plusieurs architectures basées sur une combinaison CNN + LSTM.
- Qatres modèles y sont étudiés :
    - Modèle 1 : CNN + LSTM profond : Ce modèle comporte un nombre plus élevé de couches convolutionnelles. Il offre une forte capacité de représentation mais présente un
      risque important de surapprentissage.
      
     <p align="center">
  <img width="300" height="300" alt="image"
       src="https://github.com/user-attachments/assets/f0448562-0507-43d0-aa48-825ba2fcfd54" />
  <img width="320" height="320" alt="image"
       src="https://github.com/user-attachments/assets/f7cf9ff8-6dc0-4c41-9e46-9a296b5aed81" />
      </p>


    - Modèle 2 : CNN + LSTM optimisé : Il s’agit d’une version plus légère et optimisée, qui réduit la complexité du modèle et atténue le surapprentissage, même si celui-ci reste partiellement présent.
      
     <p align="center">
  <img width="300" height="300" alt="image"
       src="https://github.com/user-attachments/assets/e2a590df-8151-4024-a473-937f6c25c04c" />
  <img width="320" height="320" alt="image"
       src="https://github.com/user-attachments/assets/97e310d0-36dd-406d-be46-8127e556b156" />
      </p>



    - Modèle 3 : CNN + LSTM dans le domaine fréquentiel : Ce modèle est adapté au domaine fréquentiel, où les signaux sont transformés par FFT avant l’apprentissage.
<p align="center">
  <img width="300" height="300" alt="image"
       src="https://github.com/user-attachments/assets/e11c6fe7-f7b5-42a6-8ae2-5c5cd039e72c" />
  <img width="300" height="300" alt="image"
       src="https://github.com/user-attachments/assets/49bc56c4-9aed-476b-9c8d-a8eb90490586" />
</p>


Toutes les informations nécessaires à l’exécution du notebook (choix du domaine temporel ou fréquentiel, prétraitement des données, normalisation, entraînement, évaluation) sont clairement indiquées avant chaque cellule, afin de faciliter la compréhension et la reproduction des résultats.

## Notebook 2
Ce notebook présente un modèle de Deep Learning hybride combinant un réseau convolutif CNN 1D, un réseau récurrent BiLSTM et un mécanisme d’attention, appliqué à la classification automatique de formes d’onde 6G à partir de signaux temporels complexes I/Q.
- Chargement des données
 -Source des données
Les signaux sont stockés dans un fichier MATLAB :

data_set_40k.mat

-Ce fichier contient :
40 000 signaux,
répartis équitablement entre 4 classes (OFDM, SC-FDM, OTFS, AFDM),
chaque signal étant composé de 4096 échantillons complexes.

Le chargement est effectué à l’aide de scipy.io.loadmat.

- Organisation des données

Après chargement : les données sont converties en tableaux NumPy,
chaque signal est restructuré sous la forme : X.shape = (N_signaux, 2, 4096)
où :
2 correspond aux canaux I et Q,
4096 à la longueur temporelle du signal.
Les labels sont convertis en entiers (int64) afin d’être compatibles avec la fonction de perte CrossEntropyLoss.

- Prétraitement des données
   -Normalisation
Les signaux sont normalisés afin de : stabiliser l’apprentissage, éviter que certaines amplitudes dominent l’optimisation, faciliter la convergence du réseau.

La normalisation est appliquée globalement sur l’ensemble du dataset.

   -Découpage du dataset

Les données sont séparées en trois sous-ensembles disjoints : 80 % pour l’entraînement, 10 % pour la validation, 10 % pour le test final.

Ce découpage permet : d’entraîner le modèle, d’ajuster les hyperparamètres sans biais, d’évaluer les performances sur des données jamais vues.

Des DataLoader PyTorch sont ensuite créés pour chaque sous-ensemble.

-Augmentation de données RF

Afin de rendre le modèle plus robuste aux conditions radio réalistes, des augmentations spécifiques aux signaux RF sont appliquées uniquement pendant l’entraînement :

rotation aléatoire de phase, décalage fréquentiel (CFO), décalage temporel circulaire, ajout de bruit AWGN avec un SNR aléatoire.

Ces augmentations simulent : les imperfections de synchronisation, le bruit du canal, la variabilité des conditions de transmission.

- Architecture du modèle

Le modèle utilisé est une architecture hybride CNN + BiLSTM + Attention, appelée :

  -Partie CNN (extraction locale)

La partie convolutionnelle 1D : traite les deux canaux I/Q, extrait des motifs locaux dans le signal, réduit progressivement la dimension temporelle.

  -Des blocs résiduels (ResBlock1D) sont utilisés pour : faciliter la propagation du gradient, améliorer la stabilité de l’apprentissage, augmenter la profondeur du réseau sans dégradation des performances.

   -Partie BiLSTM (modélisation temporelle)

La sortie du CNN est transposée puis envoyée vers un BiLSTM : il modélise les dépendances temporelles longues, il analyse l’évolution des caractéristiques extraites dans le temps, la bidirectionnalité permet d’exploiter le contexte passé et futur.
 
   -Mécanisme d’attention
    Un pooling par attention est appliqué à la sortie du BiLSTM :

le réseau apprend à pondérer les instants temporels les plus informatifs, cela permet de concentrer la décision sur les parties pertinentes du signal, la sortie est un vecteur global de caractéristiques.

   -Couche de classification

La tête du réseau est composée de : couches entièrement connectées, fonctions d’activation GELU, dropout pour la régularisation.

La sortie finale correspond aux scores de probabilité pour chaque classe de forme d’onde.

  - Entraînement du modèle
     -Fonction de perte

La fonction de perte utilisée est : CrossEntropyLoss, avec label smoothing, afin de :réduire la surconfiance du modèle, améliorer la généralisation.
    -Optimisation

Optimiseur : AdamW
Learning rate maximal : 3e-3
Scheduler : OneCycleLR

Ce choix permet : une montée progressive du learning rate, une meilleure convergence, une réduction du risque de surapprentissage.

   -Stratégie d’entraînement entraînement sur plusieurs époques, suivi des performances sur le jeu de validation, sauvegarde du meilleur modèle selon l’accuracy de validation.

Les courbes suivantes sont tracées :





 <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/5d8228e8-9653-4665-9461-9dde8fefbb4a" />

-Une matrice de confusion est calculée afin d’analyser :





 <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/94b635d7-a184-47ab-b951-18e063dea3de" />





## Suivi des performances
- Pour chaque modèle, les courbes de loss et d’accuracy sont affichées afin d’analyser le comportement de l’apprentissage.
- Ces courbes permettent :
  
      - de suivre la convergence du modèle au fil des époques
      - de détecter un éventuel surapprentissage
      - de comparer les performances entre les différentes architectures
## Comparaison 
<p align="center">
  <img width="351" height="214" alt="image"
       src="https://github.com/user-attachments/assets/4bd36dc3-8a45-489f-a91a-8acd7b68a8d1" />
</p>
