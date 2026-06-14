# World Cup Predictor 2026

Application Streamlit hors ligne pour predire les matchs de la Coupe du Monde 2026 a partir de CSV locaux.

## Donnees

Placez les fichiers Kaggle complets dans `data/` :

- `results.csv`
- `fifa_ranking.csv`
- `goalscorers.csv`
- `shootouts.csv`

Le projet inclut des CSV locaux de demonstration pour permettre un lancement immediat sans API externe. Ils peuvent etre remplaces par les datasets Kaggle complets en conservant les memes noms de fichiers.

## Lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

Au premier lancement, si `model/model.pkl` n'existe pas, le modele est entraine localement puis sauvegarde. Les lancements suivants reutilisent le fichier sauvegarde.

## Structure

```text
world-cup-predictor/
├── app.py
├── model/
│   ├── train.py
│   ├── predict.py
│   └── model.pkl
├── data/
│   ├── results.csv
│   ├── fifa_ranking.csv
│   ├── goalscorers.csv
│   └── shootouts.csv
├── utils/
│   └── helpers.py
├── requirements.txt
└── README.md
```
