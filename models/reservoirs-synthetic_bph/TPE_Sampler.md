[Algorithms for Hyper-Parameter Optimization](https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)


### Principe de la méthode Tree-structured Parzen Estimator (TPE)

Le TPE est une approche de recherche bayésienne qui fonctionne comme suit :

1. **Modélisation de l'espace des hyperparamètres :**
   - Au lieu de modéliser directement la fonction de coût (comme le feraient d'autres méthodes bayésiennes), le TPE sépare les évaluations en deux distributions distinctes :
     - Une distribution des **bons hyperparamètres** (ceux qui produisent de faibles valeurs de la fonction de coût).
     - Une distribution des **mauvais hyperparamètres** (ceux qui produisent de grandes valeurs de la fonction de coût).

2. **Formulation de deux distributions :**
   - On utilise deux distributions :
     - **\( l(x) \)** : La probabilité de trouver un point ayant une faible valeur de la fonction de coût (c'est-à-dire dans le bas du spectre des performances).
     - **\( g(x) \)** : La probabilité de trouver un point ayant une haute valeur de la fonction de coût.
   - En pratique, on définit un seuil \( \gamma \) et on divise les observations entre les bonnes (celles ayant des coûts inférieurs à ce seuil) et les mauvaises (celles ayant des coûts supérieurs).

3. **Calcul de la probabilité pour chaque échantillon :**
   - Contrairement aux méthodes classiques, le TPE calcule la **probabilité de chaque échantillon** en se basant sur ces deux distributions \( l(x) \) et \( g(x) \).
   - En particulier, l'objectif est de maximiser le rapport \( \frac{l(x)}{g(x)} \), ce qui permet de sélectionner des hyperparamètres ayant une **forte probabilité d'être parmi les bons**.

4. **Arborescence structurée (Tree-structured) :**
   - Le TPE est "tree-structured" parce qu’il peut gérer des **dépendances et relations conditionnelles entre les hyperparamètres**.
   - Cela signifie qu'il peut explorer des espaces de recherche où certains hyperparamètres dépendent d'autres (par exemple, si un certain modèle est sélectionné, alors d'autres hyperparamètres spécifiques à ce modèle peuvent être activés).
   - Cette structure hiérarchique permet de représenter des espaces de recherche complexes tout en restant efficace.

5. **Processus itératif :**
   - À chaque itération, le TPE **utilise les observations passées** (les configurations d'hyperparamètres déjà testées et leur performance) pour ajuster ses distributions \( l(x) \) et \( g(x) \).
   - Il continue à proposer de nouveaux ensembles d'hyperparamètres qui maximisent le rapport \( \frac{l(x)}{g(x)} \), en ciblant les régions de l'espace de recherche où il est plus probable de trouver des valeurs de la fonction de coût faibles.

### Pourquoi utiliser TPE ?

- **Efficacité pour les hyperparamètres discrets ou catégoriels** : Contrairement à des méthodes comme le **Gaussian Process** qui fonctionnent bien pour des espaces continus, le TPE est mieux adapté aux **espaces discrets** ou **catégoriels**, ainsi qu'aux espaces avec des dépendances entre les hyperparamètres.
  
- **Adaptabilité** : La structure arborescente permet de modéliser des relations complexes entre les hyperparamètres.

- **Approche bayésienne** : Le TPE est plus échantillon-efficient que les approches de recherche aléatoire ou de grille, car il exploite les informations des essais précédents pour proposer de nouvelles configurations.
