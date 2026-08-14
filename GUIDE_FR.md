# Guide final en français

## Ce que l'article étudie

L'article prévoit un intervalle pour le rendement du lendemain de chaque titre
d'un grand panel S&P 500. Le modèle ponctuel est une régression ridge commune.
La nouveauté est la calibration HSA-CP : elle combine trois quantiles conformes
(titre, secteur, marché), apprend leur poids par secteur, puis corrige l'erreur
de couverture sectorielle avec un contrôleur en ligne.

## Pourquoi 2024 apparaît dans l'article

L'année 2024 n'est pas la période finale annoncée comme résultat. Elle sert à
empêcher la fuite d'information :

1. janvier--mai 2024 : verrouillage de la cohorte et initialisation en ligne ;
2. juin--décembre 2024 : sélection de l'architecture et des taux ;
3. janvier--décembre 2025 : évaluation finale.

Le fait d'utiliser 2024 avant 2025 est donc nécessaire pour une expérience
chronologique. L'erreur antérieure ne venait pas de la présence de 2024, mais
de l'utilisation de la disponibilité future 2025 pour exclure des titres. Ce
filtre a été supprimé.

Plus précisément, le modèle ponctuel est réajusté jusqu'au 29 décembre 2023.
Le 2 janvier 2024 est donc la première séance produisant un résidu postérieur à
ce réajustement. Le 31 mai est la dernière séance avant le début de la
validation le 3 juin : 105 séances servent à l'initialisation, 147 à la
validation et 250 à l'évaluation 2025. Le 23 mai est la date du snapshot des
constituants, et non la fin de la calibration.

## Corrections principales

- 501 titres sont désormais verrouillés au 31 mai 2024 ; seuls GEV et SOLV
  sont exclus faute d'historique suffisant avant le verrouillage.
- Quatorze titres auparavant exclus à cause de leur indisponibilité future ont
  été réintégrés.
- Le quantile conforme utilise exactement le rang
  `ceil((n+1)(1-alpha))`, sans décalage d'indice.
- Le rendement sectoriel explicatif exclut le titre cible.
- Le plancher de volatilité et la standardisation utilisent uniquement
  l'entraînement.
- La pénalité ridge 100 est choisie avec 2023 seulement.
- Les taux ACI (.005), PControl (.05), Hedge (.04) et contrôleur HSA (.05) sont
  déterminés avec la validation 2024.
- ALCE20 (niveau titre) et XLCE20 (niveau transversal) sont séparés.
- L'inférence utilise Newey--West, Holm et 2 000 bootstraps par blocs de 5, 10
  et 20 jours.

## Théorèmes et preuves ajoutés

L'article contient maintenant :

1. un théorème de regret Hedge pour la perte normalisée et tronquée réellement
   utilisée par le code ;
2. une proposition donnant l'identité exacte de calibration sectorielle ;
3. un corollaire fini en T bornant l'écart moyen de couverture ;
4. les preuves des trois résultats.

Deux pseudocodes formels ont également été ajoutés : l'algorithme quotidien
HSA-CP et le protocole complet de sélection chronologique sans fuite.

L'identité a aussi été vérifiée numériquement pour les 11 secteurs : l'erreur
maximale entre les deux côtés est inférieure à 2.1e-17. Ces résultats ne sont
pas présentés comme une garantie de couverture conditionnelle exacte pour des
rendements dépendants.

## Résultat final à citer

HSA-CP obtient une couverture de 90,0455 %, un score d'intervalle de 0,096016,
ALCE20 de 0,047270 et XLCE20 de 0,017636 sur 121 744 intervalles valides. Les
couvertures sectorielles vont de 89,7662 % à 90,4000 %. L'amélioration est
robuste à toutes les longueurs de blocs face à RegimeCP, ACI, RollingCP et
GraphCP. Elle ne l'est pas face à SplitCP, PControl et SectorCP ; l'article le
dit explicitement.

Sans réajustement des taux, les couvertures aux niveaux nominaux 80 %, 90 % et
95 % sont respectivement 79,9588 %, 90,0455 % et 95,1883 %. L'ablation montre
que l'échelle EWM-21 principale donne le meilleur score d'intervalle. Les
seuils de disponibilité 98 % et 99 % et les dates de verrouillage 30 avril,
23 mai et 31 mai sélectionnent tous la même cohorte de 501 titres.

## Compilation Overleaf

1. Ouvrir le modèle officiel IEEE Conference dans Overleaf.
2. Remplacer le fichier principal par `paper.tex`.
3. Importer `references.bib` et le dossier `figures`.
4. Choisir pdfLaTeX et compiler.
5. Vérifier le nombre de pages et lancer ensuite IEEE PDF eXpress si la
   conférence le demande.

La prévisualisation locale compte 8 pages, références comprises. Il reste donc
une marge sous la limite de 10 pages, même si la pagination officielle Overleaf
varie légèrement.

## Règle IEEE BigData 2026

La conférence visée est IEEE BigData 2026, et non ICDM. Le CFP officiel indique
une évaluation **single-blind** : il faut conserver le nom et l'affiliation dans
la version soumise. Ne pas remplacer l'autrice par « Anonymous ». Les articles
réguliers complets utilisent le format IEEE deux colonnes et ne doivent pas
dépasser 10 pages, références comprises.

## GitHub

Le paquet est prêt à être placé dans un dépôt GitHub, mais aucun faux lien n'a
été inventé. Après création du vrai dépôt, insérer son URL dans le paragraphe
de reproductibilité et recompiler. Vérifier les conditions du fournisseur avant
de rendre publiques les données ou les rendements dérivés.
