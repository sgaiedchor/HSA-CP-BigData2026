# Publication du dépôt HSA-CP sur GitHub

## 1. Créer le dépôt

1. Connectez-vous à https://github.com.
2. Cliquez sur le signe `+`, puis sur **New repository**.
3. Saisissez le nom `hsa-cp`.
4. Ajoutez la description : `Code, data, and reproducibility materials for HSA-CP.`
5. Choisissez **Public**.
6. Ne cochez pas l'ajout automatique d'un README, d'une licence ou d'un
   fichier `.gitignore`, car ils figurent déjà dans ce paquet.
7. Cliquez sur **Create repository**.

## 2. Téléverser les fichiers sans ligne de commande

1. Dans le nouveau dépôt, cliquez sur **uploading an existing file**.
2. Décompressez d'abord le paquet reçu.
3. Faites glisser le contenu du dossier décompressé dans la zone GitHub. Ne
   téléversez pas uniquement le fichier ZIP.
4. Vérifiez que les deux classeurs Excel, les scripts et les dossiers
   `results`, `tables` et `figures` apparaissent dans la liste.
5. Dans **Commit changes**, écrivez :
   `Initial public reproducibility release`.
6. Cliquez sur **Commit changes**.

## 3. Créer une version stable

1. Ouvrez **Releases**, puis **Draft a new release**.
2. Créez le tag `v1.0.0`.
3. Donnez à la version le titre `HSA-CP reproducibility release v1.0.0`.
4. Résumez le contenu : code, données, résultats, figures et sources LaTeX.
5. Cliquez sur **Publish release**.

## 4. Archiver sur Zenodo

1. Connectez Zenodo à votre compte GitHub.
2. Activez l'archivage du dépôt `hsa-cp`.
3. Publiez ou réarchivez la release `v1.0.0`.
4. Copiez le DOI attribué par Zenodo.
5. Ajoutez ce DOI dans `CITATION.cff`, le README et la déclaration de
   disponibilité du manuscrit.

## 5. Ajouter le lien au manuscrit

Après la création réelle du dépôt, remplacez `VOTRE-COMPTE` par votre nom de
compte GitHub :

```latex
\subsection*{Data and Code Availability}
The data, implementation, experimental configuration, aggregate results, and
reproducibility materials are publicly available at
\url{https://github.com/VOTRE-COMPTE/hsa-cp}. A versioned archival copy is
available at \url{https://doi.org/DOI-ZENODO}.
```

N'insérez jamais un lien fictif ou un DOI provisoire dans la version finale.
