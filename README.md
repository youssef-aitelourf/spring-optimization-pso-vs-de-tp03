# Projet personnel - Comparaison PSO vs DE

Ce dépôt présente une étude comparative de deux métaheuristiques d'optimisation continue appliquées à un **problème de conception de ressort hélicoïdal sous contraintes** :

- `PSO` (Particle Swarm Optimization), avec topologie `ring` ou `global`
- `DE/rand/1/bin` (Differential Evolution)

Ce projet personnel peut etre compris et execute facilement par une personne externe. L'objectif est de **reproduire des experiences**, **comparer les performances des algorithmes** et **generer automatiquement des resultats exploitables** sous forme de tableaux, historiques et graphiques.

## Aperçu rapide

En lançant le pipeline principal, le projet :

- exécute plusieurs runs Monte-Carlo pour `PSO` et `DE`
- enregistre l'historique de convergence de chaque run
- calcule des statistiques finales comparatives
- génère plusieurs graphiques d'analyse
- sauvegarde tous les résultats dans un dossier de sortie

Le point d'entrée principal est `main.py`.

## Problème étudié

Le problème consiste à optimiser les paramètres d'un ressort hélicoïdal en respectant des contraintes de faisabilité. Chaque solution candidate est évaluée via :

- une **fonction objectif**
- des **contraintes**
- une **pénalisation** des violations pour permettre aux algorithmes de comparer des solutions non faisables

Cette logique est centralisée dans `problem.py`.

## Prérequis

- Python `3.11+` recommandé
- `pip`
- optionnel : `python-docx` si vous souhaitez régénérer le rapport Word

## Installation

```bash
git clone <url-du-depot>
cd <nom-du-depot>

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
pip install python-docx        # uniquement pour générer le rapport Word
```

## Lancement rapide

Pour vérifier rapidement que le projet fonctionne :

```bash
python main.py
```

Cette commande lance les expériences avec les paramètres par défaut et écrit les résultats dans le dossier `outputs/`.

## Reproduire les resultats de reference

La commande suivante correspond a une configuration de reference recommandee :

```bash
python main.py \
  --runs 50 \
  --max-iter 800 \
  --stagnation-patience 9999 \
  --seed 2026 \
  --penalty-coeff 1e6 \
  --swarm-size 40 \
  --de-population-size 40 \
  --pso-neighborhood ring \
  --de-f 0.7 \
  --de-cr 0.9 \
  --tp1-best-cost 0.012806 \
  --output-dir Rapport/results_report
```

`--stagnation-patience 9999` désactive en pratique l'arrêt anticipé par stagnation. Cela garantit un budget d'évaluation identique pour `PSO` et `DE`, ce qui rend la comparaison plus rigoureuse.

## Ce que le script produit

À la fin d'une exécution, `main.py` effectue les étapes suivantes :

1. lance les expériences Monte-Carlo
2. sauvegarde les sorties brutes
3. calcule les statistiques finales
4. génère les figures
5. affiche les chemins des fichiers produits

## Fichiers générés

Le dossier de sortie contient typiquement :

| Fichier | Description |
|---------|-------------|
| `runs_history.csv` / `runs_history.pkl` | Historique complet par run, itération, budget et indicateurs |
| `final_statistics.csv` | Résumé statistique final par algorithme |
| `budget_comparability.csv` | Vérification de la comparabilité des budgets d'évaluation |
| `convergence_profile.csv` | Profil de convergence normalisé |
| `tp1_comparison.csv` | Comparaison avec une valeur de reference externe |
| `convergence_pso_vs_de.png` | Courbes de convergence comparées |
| `boxplot_final_costs.png` | Distribution finale des coûts |
| `violin_final_costs.png` | Visualisation fine des distributions finales |
| `diversity_evolution.png` | Évolution de la diversité des populations |
| `best_run_convergence.png` | Courbe du meilleur run pour chaque algorithme |
| `feasibility_rate.png` | Taux de solutions faisables au fil du budget |

## Paramètres utiles

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `--runs` | Nombre de runs Monte-Carlo par algorithme | `50` |
| `--max-iter` | Nombre maximal d'itérations | `800` |
| `--stagnation-patience` | Arrêt après `N` itérations sans amélioration | `120` |
| `--stagnation-epsilon` | Amélioration minimale considérée comme significative | `1e-8` |
| `--seed` | Graine aléatoire de base | `2026` |
| `--penalty-coeff` | Intensité de la pénalisation des contraintes | `1e6` |
| `--swarm-size` | Taille de l'essaim PSO | `40` |
| `--de-population-size` | Taille de la population DE | `40` |
| `--pso-neighborhood` | Topologie PSO : `ring` ou `global` | `ring` |
| `--pso-inertia` | Poids d'inertie de PSO | `0.72` |
| `--pso-c1` | Coefficient cognitif de PSO | `1.49` |
| `--pso-c2` | Coefficient social de PSO | `1.49` |
| `--de-f` | Facteur différentiel de DE | `0.7` |
| `--de-cr` | Taux de croisement de DE | `0.9` |
| `--tp1-best-cost` | Référence de comparaison externe optionnelle | `None` |
| `--output-dir` | Dossier dans lequel écrire les résultats | `outputs` |

## Structure du projet

| Fichier | Rôle |
|---------|------|
| `main.py` | Point d'entrée CLI et orchestration complète |
| `problem.py` | Définition du problème, contraintes, bornes et pénalisation |
| `algorithms.py` | Implémentation de `PSO` et `DE/rand/1/bin` |
| `experiments.py` | Boucle d'expériences Monte-Carlo et sauvegarde des sorties |
| `analysis.py` | Calcul des statistiques et génération des figures |
| `Rapport/generate_report_docx.py` | Generation du document Word final |

## Generation du document Word

Si les fichiers de résultats attendus sont disponibles, vous pouvez générer le document Word avec :

```bash
python Rapport/generate_report_docx.py
```

Le document est sauvegarde dans `Rapport/Rapport_TP03_8INF852_AITELOURF_final.docx`.

## À qui s'adresse ce dépôt ?

Ce dépôt peut servir à :

- une personne qui veut executer rapidement un projet Python personnel
- un lecteur externe qui veut comprendre la comparaison `PSO` vs `DE`
- une personne souhaitant réutiliser le pipeline expérimental pour d'autres comparaisons

## Résumé

Si vous ne devez retenir qu'une seule commande pour démarrer :

```bash
python main.py
```

Et si vous voulez reproduire les resultats de reference, utilisez la commande longue fournie plus haut dans la section dediee.
