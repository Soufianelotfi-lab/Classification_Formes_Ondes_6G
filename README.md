# Classification_Formes_Ondes_6G
## Présentation du projet
Ce projet a pour objectif de classifier différentes formes d’ondes 6G à partir de signaux I/Q, en s’appuyant sur des techniques de Deep Learning. Les signaux étudiés correspondent à différentes modulations et formes d’ondes, dont les caractéristiques peuvent être difficiles à distinguer par des méthodes classiques.

Les données sont analysées selon deux représentations complémentaires. Dans le domaine temporel, le modèle exploite directement l’évolution des échantillons I/Q afin d’apprendre les dynamiques temporelles propres à chaque forme d’onde. Dans le domaine fréquentiel, les signaux sont transformés à l’aide de la FFT afin de mettre en évidence leur contenu spectral, ce qui permet de capturer des signatures fréquentielles caractéristiques.

## Organisation du projet
Le projet est structuré autour de deux notebooks principaux, chacun correspondant à une approche de modélisation.
#### Notebook 1 – Modèles CNN + LSTM
- Ce premier notebook regroupe plusieurs architectures basées sur une combinaison CNN + LSTM.
- Trois modèles y sont étudiés :
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
