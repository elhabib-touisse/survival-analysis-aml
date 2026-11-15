# Apprentissage Statistique Supervisé en Analyse de Survie

[![Notebook HTML](https://img.shields.io/badge/Notebook_Interactif-Version_HTML-1d4ed8?style=flat-square)](https://htmlpreview.github.io/?https://github.com/elhabib-touisse/survival-analysis-aml/blob/main/analyse_survie_leucemie_aigue.html)
[![Notebook Jupyter](https://img.shields.io/badge/Code_Source-Notebook_.ipynb-f97316?style=flat-square)](analyse_survie_leucemie_aigue.ipynb)

- **Accès direct au Notebook rendu en HTML** : [Ouvrir le Notebook HTML](https://htmlpreview.github.io/?https://github.com/elhabib-touisse/survival-analysis-aml/blob/main/analyse_survie_leucemie_aigue.html) *(ou via [GitHub Pages](https://elhabib-touisse.github.io/survival-analysis-aml/analyse_survie_leucemie_aigue.html))*
- **Fichier Jupyter Notebook** : [`analyse_survie_leucemie_aigue.ipynb`](analyse_survie_leucemie_aigue.ipynb)

---

## Formulation du problème

Soit $(T_i, C_i, X_i)_{1 \le i \le n}$ un échantillon i.i.d. où $T_i > 0$ désigne la durée de survie réelle, $C_i > 0$ la variable de censure à droite et $X_i \in \mathbb R^p$ le vecteur de covariables. En raison de la censure, on observe uniquement le triplet $(\tilde T_i, \Delta_i, X_i)$ défini par :

```math
\tilde{T}_i = \min(T_i, C_i), \qquad \Delta_i = \mathbf{1}(T_i \le C_i) \in \{0, 1\}
```

Sous l'hypothèse de censure non informative conditionnellement aux covariables, l'objectif est d'estimer la fonction de survie conditionnelle $S(t \mid x) = \mathbb P(T > t \mid X = x)$ ou la fonction de risque instantané (*hazard rate*) :

```math
h(t \mid x) = \lim_{dt \to 0^+} \frac{\mathbb{P}(t \le T < t + dt \mid T \ge t, X = x)}{dt}, \qquad S(t \mid x) = \exp\left( -\int_0^t h(u \mid x)\,du \right)
```

Dans cette étude ($n = 3173$ patients atteints de Leucémie Myéloïde Aiguë), chaque patient $i$ est décrit par un vecteur clinico-cytogénétique $Z_i \in \mathbb R^{12}$ (6 biomarqueurs hématologiques et 6 indicateurs ISCN) et par le multiensemble de ses mutations somatiques de cardinalité variable $K_i$ :

```math
\mathcal{M}_i = \bigl\{ (g_{i,k}, v_{i,k}, e_{i,k}) \bigr\}_{k=1}^{K_i}
```

où $g \in \mathcal G$ désigne le gène muté, $v \in [0, 1]$ la fréquence allélique du variant (VAF) et $e \in \mathcal E$ la classe d'effet fonctionnel. On construit une application d'agrégation :

```math
\Phi : \mathcal{P}_{\mathrm{fini}}(\mathcal{X}_{\mathrm{mut}}) \to \mathbb{R}^{61}
```

telle que le vecteur de régression complet du patient $i$ soit :

```math
X_i = \left( Z_i^\top, \Phi(\mathcal{M}_i)^\top \right)^\top \in \mathbb{R}^{73}
```

---

## Ce qui est développé dans le Notebook

L'ensemble de la démarche mathématique et empirique est présenté dans le [Notebook](https://htmlpreview.github.io/?https://github.com/elhabib-touisse/survival-analysis-aml/blob/main/analyse_survie_leucemie_aigue.html) :

1. **Analyse non-paramétrique et effet centre** : estimateurs de Kaplan–Meier $\hat S_{\mathrm{KM}}(t)$, de Nelson–Aalen $\hat H_{\mathrm{NA}}(t)$ et test du Log-Rank multivarié sur les 23 centres hospitaliers.
2. **Construction de l'opérateur d'agrégation moléculaire** $\Phi(\mathcal M_i)$ : indicatrices des gènes récurrents, statistiques de la charge clonale VAF (maximum, moyenne), regroupement des gènes rares et décompte par classe d'effet fonctionnel.
3. **Validation croisée stratifiée 5-Fold sans fuite d'information** : sur les $80\%$ d'entraînement ($n = 2538$), imputation, standardisation et maillages temporels de PC-Hazard sont estimés strictement dans chaque pli ; chaque modèle retenu est ré-entraîné sur l'ensemble d'entraînement complet puis évalué sur le jeu de test externe ($20\%$).
4. **Modèle de Cox à risques proportionnels et Test de Schoenfeld** : maximisation de la vraisemblance partielle pénalisée, estimateur de Breslow $\hat H_0(t)$ et test d'indépendance temporelle des résidus de Schoenfeld échelonnés. Le rejet de l'hypothèse des risques proportionnels sur 7 covariables (dont `MUT_NPM1`, `PLT`, `MUT_U2AF1`, `MUT_TP53`, `MUT_WT1`, `MUT_SF3B1`) motive le passage à des modèles non-linéaires et non-proportionnels.
5. **Arbres de survie, ensembles et réseaux de neurones en temps continu** :
   - **Arbre de survie (CART)** et **Random Survival Forest (RSF)** (statistique du Log-Rank + estimateurs de Nelson–Aalen locaux),
   - **Gradient Boosting de Cox** sur les pseudo-résidus de la vraisemblance partielle,
   - **DeepSurv** (log-risque relatif neuronal sous risques proportionnels),
   - **PC-Hazard (*Piecewise Constant Hazard*)** : risque constant par morceaux $\lambda_j(x)$ entraîné par log-vraisemblance continue exacte sur **grille équidistante grossière** ($L=5$) et **grille aux quantiles de Kaplan–Meier** ($L=20$).
6. **Évaluation et Calibration** : C-index de Harrell, C-index pondéré par IPCW de Uno ($\tau = 7$ ans), *Integrated Brier Score* (IBS) et courbes de calibration par quintiles de Kaplan–Meier à l'horizon médian $t^\star = 1.38$ an.

---

## Résultats (Validation Croisée 5-Fold & Test Externe)

### 1. Validation croisée stratifiée 5-Fold sur l'entraînement ($n = 2538$, Moyenne $\pm$ Écart-type)

| Modèle | CV Harrell | CV Uno IPCW | CV IBS |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting de Cox** | 0.7383 ± 0.0111 | **0.7154 ± 0.0142** | **0.1634 ± 0.0058** |
| **Random Survival Forest (RSF)** | **0.7388 ± 0.0131** | 0.7146 ± 0.0163 | 0.1646 ± 0.0065 |
| **Modèle de Cox (Benchmark)** | 0.7252 ± 0.0119 | 0.7021 ± 0.0170 | 0.1677 ± 0.0056 |
| **DeepSurv** | 0.7231 ± 0.0162 | 0.7002 ± 0.0205 | 0.1690 ± 0.0068 |
| **PC-Hazard (Grille Kaplan–Meier, L = 20)** | 0.7254 ± 0.0121 | 0.7000 ± 0.0190 | 0.1688 ± 0.0067 |
| **PC-Hazard (Grille grossière, L = 5)** | 0.7197 ± 0.0161 | 0.6949 ± 0.0216 | 0.1720 ± 0.0066 |
| **Arbre de Survie (Log-Rank)** | 0.6895 ± 0.0147 | 0.6701 ± 0.0188 | 0.1838 ± 0.0084 |

### 2. Performance finale sur l'ensemble de Test externe ($n_{\mathrm{test}} = 631$)

| Modèle | Test Harrell | Test Uno IPCW | Test IBS | MAE Calibration ($t^\star$) |
| :--- | :---: | :---: | :---: | :---: |
| **DeepSurv** | 0.7294 | **0.7140** | 0.1668 | 0.0455 |
| **Random Survival Forest (RSF)** | **0.7366** | 0.7130 | 0.1689 | 0.0559 |
| **PC-Hazard (Grille Kaplan–Meier, L = 20)** | 0.7356 | 0.7129 | **0.1640** | 0.0295 |
| **Gradient Boosting de Cox** | 0.7363 | 0.7123 | 0.1666 | 0.0301 |
| **PC-Hazard (Grille grossière, L = 5)** | 0.7329 | 0.7121 | **0.1640** | **0.0229** |
| **Modèle de Cox (Benchmark)** | 0.7284 | 0.7089 | 0.1670 | 0.0285 |
| **Arbre de Survie (Log-Rank)** | 0.6955 | 0.6616 | 0.1887 | 0.0392 |

---

## Données

Les données utilisées dans ce notebook correspondent au challenge **QRT × Institut Gustave Roussy** (*Leukemia Survival Prediction*), disponible sur le site [Challenge Data de l'ENS](https://challengedata.ens.fr/).

---

## Références et lectures pour approfondir

Si le sujet vous intéresse, vous pouvez consulter :
- **Livre de référence** : [Machine Learning for Survival Analysis (`mlsabook.com`)](https://www.mlsabook.com/)
- **Revue et fondements récents** : [arXiv:2410.01086](https://arxiv.org/abs/2410.01086)
- **Sur PC-Hazard et le choix de la grille de discrétisation temporelle (équidistante vs quantiles de Kaplan–Meier)** : H. Kvamme & Ø. Borgan, *Continuous and Discrete-Time Survival Prediction with Neural Networks*, [arXiv:1910.06724](https://arxiv.org/abs/1910.06724)
