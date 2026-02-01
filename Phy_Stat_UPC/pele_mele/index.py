# %% [markdown]
# ---
# title: Pèle mêle sur les probabilités 
# description: "Quelques exemples de visualisations de résultats élémentaires en théorie des probabilités."
# author: Sacha Guerrini
# date: today
# format: html
# jupyter: python3
# number-figures: true
# number-sections: true
# execute:
#   echo: false
# ---

# %%
#| output: false
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(
    "./paper.mplstyle"
)

sns.color_palette("deep")

# %% [markdown]
"""
# Echantillonage de la loi binomiale

Dans ce notebook, nous allons utiliser la librairie `scipy.stats` pour discuter quelques résultats élémentaires en théorie des probabilités que nous avons étudiés.

## Loi binomiale

Echantillonons la loi binomiale avec des paramètres $n=10$ et $p=0.5$.
"""

# %%
#| label: fig-binom-10-05
#| fig-cap: Distribution binomiale avec $n=10$ et $p=0.5$
#| fig-alt: Histogramme des échantillons et fonction de masse de probabilité de la distribution binomiale.
# Sample from the binomial distribution
binom_samples = stats.binom.rvs(n=10, p=0.5, size=2000)

# Probability mass function of the binomial distribution
k = np.arange(0, 11)
pmf = stats.binom.pmf(k, n=10, p=0.5)

# Plotting the histogram of samples and the PMF
plt.figure(figsize=(8, 6))

plt.hist(binom_samples, bins=np.arange(-0.5, 11.5, 1), density=True, alpha=0.6, color='g', label='Histogram of samples')
plt.plot(k, pmf, 'bo', ms=8, label='PMF', color='red')
plt.vlines(k, 0, pmf, colors='r', lw=5, alpha=0.5)

plt.title(r'Binomial Distribution: $n=10$, $p=0.5$')
plt.xlabel('Number of successes')
plt.xticks(k)
plt.ylabel('Probability')
plt.legend()
plt.show()

# %% [markdown]

"""
On voit que la distribution est en très bon accord avec la distribution théorique, ici représentée en rouge.

Essayons d'augmenter le nombre de répétitions de l'expérience, $n$.
"""

# %%
# | label: fig-binom-100-05
# | fig-cap: Distribution binomiale avec $n=100$ et $p=0.5$
# | fig-alt: Histogramme des échantillons et fonction de masse de probabilité de la distribution binomiale.

# Sample from the binomial distribution
binom_samples = stats.binom.rvs(n=100, p=0.5, size=2000)

# Probability mass function of the binomial distribution
k = np.arange(0, 101)
pmf = stats.binom.pmf(k, n=100, p=0.5)

# Plotting the histogram of samples and the PMF
plt.figure(figsize=(8, 6))

plt.plot(k, pmf, 'bo', ms=8, label='PMF', color='red')

plt.hist(binom_samples, bins=np.arange(-0.5, 101.5, 1), density=True, alpha=0.6, color='g', label='Histogram of samples')

plt.title(r'Binomial Distribution: $n=10$, $p=0.5$')
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.legend()
plt.show()

# %% [markdown]

"""
Excellent, on comprend que la distribution de probabilité théorique correspond bien à la distribution empirique obtenue par échantillonnage. Regardons ce qu'il se passe si l'on varie $p$.
"""

# %%
# | label: fig-binom-100-03
# | fig-cap: Distribution binomiale avec $n=100$ et $p=0.3$
# | fig-alt: Histogramme des échantillons et fonction de masse de probabilité de la distribution binomiale.

# Sample from the binomial distribution
binom_samples = stats.binom.rvs(n=100, p=0.3, size=2000)

# Probability mass function of the binomial distribution
k = np.arange(0, 101)
pmf = stats.binom.pmf(k, n=100, p=0.3)
# Plotting the histogram of samples and the PMF
plt.figure(figsize=(8, 6))

plt.plot(k, pmf, 'bo', ms=8, label='PMF', color='red')

plt.hist(binom_samples, bins=np.arange(-0.5, 101.5, 1), density=True, alpha=0.6, color='g', label='Histogram of samples')

plt.title(r'Binomial Distribution: $n=100$, $p=0.3$')
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.legend()
plt.show()

# %% [markdown]

"""
On voit que la distribution des échantillons s'est déplacée vers la gauche. On pouvait s'y attendre puisque la probabilité de succès $p$ a diminué. Ainsi, la probabilité de réaliser un grand nombre de succès parmi les $n=100$ expériences diminue également.

Enfin, on peut vérifier que la moyenne et la variance des échantillons correspondent bien aux valeurs théoriques données par :
$$
\mathbb{E}[X] = n p, \quad
\mathbb{V}[X] = n p (1-p).
$$
"""

# %%
# | label: fig-binom-100-03-mean-var
# | fig-cap: Distribution binomiale avec $n=100$ et $p=0.3$, avec indication de la moyenne théorique et empirique.
# | fig-alt: Histogramme des échantillons et fonction de masse de probabilité de la distribution binomiale, avec indication de la moyenne théorique et empirique.
# Sample from the binomial distribution
binom_samples = stats.binom.rvs(n=100, p=0.3, size=2000)

# Probability mass function of the binomial distribution
k = np.arange(0, 101)
pmf = stats.binom.pmf(k, n=100, p=0.3)
# Plotting the histogram of samples and the PMF
plt.figure(figsize=(8, 6))

plt.plot(k, pmf, 'bo', ms=8, label='PMF', color='red')
plt.axvline(100*0.3, color='black', linestyle='dashed', linewidth=1, label='Theoretical Mean')
plt.axvline(np.mean(binom_samples), color='blue', linestyle='-.', linewidth=1, label='Sample Mean')
plt.hist(binom_samples, bins=np.arange(-0.5, 101.5, 1), density=True, alpha=0.6, color='g', label='Histogram of samples')

plt.title(r'Binomial Distribution: $n=100$, $p=0.3$')
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.legend()
plt.show()

# %%
print(f"Theoretical Mean: {100*0.3}, Sample Mean: {np.mean(binom_samples)}")
print(f"Difference in Mean: {abs(100*0.3 - np.mean(binom_samples))}")

# %%
print(f"Theoretical variance: {100*0.3*0.7}, Sample Variance: {np.var(binom_samples)}")
print(f"Difference in Variance: {abs(100*0.3*0.7 - np.var(binom_samples))}")

# %% [markdown]

"""
A nouveau cela fonctionne très bien ! Et on voit que la largeur de la distribution est de l'ordre de la vingtaine d'échantillons autour de la moyenne.

## Loi binomiale à partir de loi de Bernoulli

Nous avons vu que la loi binomiale pouvait être vue comme la somme de variables aléatoires indépendantes suivant une loi de Bernoulli. 

$$
X = \sum_{i=1}^n Y_i, \quad Y_i \hookrightarrow \mathcal{B}(p)
$$
"""

# %%
# | echo: true

n = 10
p = 0.5

def draw_bernoulli(n, p):
    return stats.bernoulli.rvs(p, size=n)

# Sample from the Bernoulli distribution and sum to get Binomial samples
np.random.seed(42) # Fix the seed for reproducibility
bernoulli_samples = draw_bernoulli(n, p)
print(bernoulli_samples)


# %% [markdown]

"""
Ici, on a tiré $n=10$ échantillons d'une loi de Bernoulli de paramètre $p=0.5$. Cela correspond à tirer une configuration de succès et d'échecs pour 10 expériences indépendantes et identiques dans le contexte de la loi de Bernoulli. Dans l'exemple ci-dessus, on a obtenu 6 succès (1) et 4 échecs (0).
"""

# %%
# | echo: true
np.random.seed(100)
bernoulli_samples = draw_bernoulli(n, p)
print(bernoulli_samples)

# %% [markdown]
"""
Cette fois on a obtenu 5 succès et 5 échecs. En répétant cette expérience un grand nombre de fois, on peut reconstituer la loi binomiale.
"""

# %%
# | echo: true
num_samples = 2000

binom_from_bernoulli_samples = np.array([
    np.sum(draw_bernoulli(n, p)) for _ in range(num_samples)
])
binom_scipy_samples = stats.binom.rvs(n=n, p=p, size=num_samples)
k = np.arange(0, n+1)
pmf = stats.binom.pmf(k, n=n, p=p)

# %%
#  | label: fig-binom-bernoulli
#  | fig-cap: Comparaison entre échantillonnage de la loi binomiale via la somme de lois de Bernoulli et échantillonnage direct via `scipy.stats.binom`.
#  | fig-alt: Histogramme des échantillons obtenus par somme de lois de Bernoulli et par échantillonnage direct de la loi binomiale, avec la fonction de masse de probabilité de la distribution binomiale.
# Plotting the histogram of samples and the PMF
plt.figure(figsize=(8, 6))

plt.hist(binom_from_bernoulli_samples, bins=np.arange(-0.5, 11.5, 1), density=True, label='Samples from Bernoulli sums', histtype='step')
plt.plot(k, pmf, 'bo', ms=8, label='PMF')
plt.hist(binom_scipy_samples, bins=np.arange(-0.5, 11.5, 1), density=True, label='Samples from `scipy.stats`', histtype='step')

plt.legend(bbox_to_anchor=(0.55, -0.15), loc='upper center', frameon=False, ncols=2)
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.title('Comparison of Binomial Sampling Methods')
plt.show()

# %% [markdown]
"""
On voit quà l'erreur d'échantillonnage près, les deux méthodes d'échantillonnage de la loi binomiale sont équivalentes. 🚀

# Lien entre loi de probabilité et moments

Nous avons vu dans le cours comment calculer la moyenne et la variance d'une variable aléatoire à partir de sa loi de probabilité. On parle des moments d'une variable aléatoire pour désigner les quantités

$$
\mathbb{E}[X^k] = \sum_x x^k p_X(x).
$$

Dans le cas de la loi binomiale, nous avons calculé sa moyenne et sa variance. Il est évident qu'étant donné la loi de probabilité ($\mathbb{P}(X=k)$), on peut calculer n'importe quel moment d'une variable aléatoire. Mais il faut faire attention au fait que l'inverse n'est pas forcément vrai: deux variables aléatoires peuvent avoir la même moyenne et variance, mais des lois de probabilité différentes. Cette section vise à illustrer ce point.

Dans ce qui va suivre, nous allons considérer une variable aléatoire continue $X$ suivant une loi normale (Gaussienne) centrée réduite, i.e. de moyenne 0 et de variance 1.

$$
X \hookrightarrow \mathcal{N}(0, 1).
$$

La densité de probabilité de cette variable aléatoire est donnée par:

$$
f(x) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^2}{2}}.
$$

La probabilité que la variable aléatoire $X$ soit à valeur dans $[x_0, x_1]$ est donnée par l'intégrale:
$$
\mathbb{P}(x_0 \leq X \leq x_1) = \int_{x_0}^{x_1} f(x) dx.
$$
"""

# %%
# | label: fig-normal-0-1
# | fig-cap: Densité de probabilité de la loi normale centrée réduite. La zone rouge indique la probabilité que la variable aléatoire $X$ soit dans l'intervalle $[x_0, x_1]$. Ici, on a pris $x_0=-1$ et $x_1=1$.
# | fig-alt: Densité de probabilité de la loi normale centrée réduite
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, loc=0, scale=1)

x_0, x_1, = -1, 1
# Plotting the PDF

plt.figure(figsize=(8, 6))

plt.plot(x, pdf, lw=2, label=r'PDF $f(x)$', c='r')
plt.fill_between(x, 0, pdf, where=(x >= x_0) & (x <= x_1), color='red', alpha=0.3, label=rf'Area: $\mathbf{{P}}({x_0} \leq X \leq {x_1})$')

plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.title("Normal Distribution: $\mathcal{N}(0, 1)$")
plt.legend()
plt.show()

# %%
# | output: true
moments = [stats.norm.moment(n, loc=0, scale=1) for n in range(10)]
for i, moment in enumerate(moments):
    print(f"Moment of order {i}: {moment}")

# %% [markdown]

"""
Les moments sont informatifs sur une distribution. Par exemple, une distribution symétrique autour de zéro aura tous ses moments d'ordre impair nuls. Connaitre tous les moments d'une distribution ne permet pas toujours de reconstruire la distribution initiale cependant, mais cela est possible dans certains hypothèses. (Voir page Wikipedia, problème des moments).

Pour illustrer l'importance des moments, nous allons perturber la distribution Gaussienne centrée réduite de façon à obtenir une nouvelle distribution ayant la même moyenne et variance, mais des moments d'ordre supérieur différents.
"""

# %%
# | echo : true
gamma_1 = 0.3
perturb_pdf = pdf * (1 + gamma_1 * (x**3 - 3*x) / 6) #Expansion d'Edgeworth au premier ordre
perturb_pdf /= np.trapz(perturb_pdf, x)  # Normalisation

# %%
# | label: fig-normal-perturbed
# | fig-cap: Densité de probabilité de la loi normale centrée réduite perturbée. On a conservé la moyenne et la variance, mais les moments d'ordre supérieur sont différents.
# | fig-alt: Densité de probabilité de la loi normale centrée réduite perturbée.
plt.figure(figsize=(8, 6))

plt.plot(x, pdf, lw=2, label=r'PDF $\mathcal{N}(0,1)$', c='k', ls='--', alpha=0.5)
plt.plot(x, perturb_pdf, lw=2, label=r'Perturbed PDF', c='b')

plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")

plt.legend()
plt.show()
# %%
pertur_moments = [
    np.trapz(x**n * perturb_pdf, x) for n in range(10)
]
for i, moment in enumerate(pertur_moments):
    print(f"Perturbed Moment of order {i}: {moment}")

# %% [markdown]

"""
Il est important de noter que la perturbation faite ici conserve la moeyenne et la variance mais introduit une asymétrie dans la distribution, ce qui modifie les moments d'ordre supérieur. 

La courbe bleue n'est pas exactement une distribution car nous avons utiliser une approximation d'Edgeworth au premier ordre pour la construire. Cependant, elle illustre bien le fait que deux distributions peuvent partager les mêmes moyenne et variance tout en ayant des formes différentes, mises en évidence par leurs moments d'ordre supérieur.

# Equation de diffusion

## Marche aléatoire simple

On considère une particule effectuant une marche aléatoire simple en une dimension. À chaque pas de temps, la particule se déplace soit d'une unité vers la droite, soit d'une unité vers la gauche, avec une probabilité égale de 0.5 pour chaque direction.

Visualisons d'abord quelques trajectoires possibles de cette particule.
"""

# %%
# | echo: True
# In out example we use a step size a=1 and a time step tau=1
def draw_trajectory(num_steps, p=0.5):
    steps = np.random.choice([-1, 1], size=num_steps, p=[1-p, p]) # p is the probability to move to the right
    trajectory = np.cumsum(steps)
    return trajectory

# %%
# | label: fig-random-walk-trajectories
# | fig-cap: Quelques trajectoires possibles d'une particule effectuant une marche aléatoire simple en une dimension.
# | fig-alt: Trajectoires de marche aléatoire simple
num_trajectories = 5
num_steps = 1_000

plt.figure(figsize=(8, 6))

for _ in range(num_trajectories):
    trajectory = draw_trajectory(num_steps)
    plt.plot(trajectory, range(num_steps), alpha=0.7)

plt.title("Random Walk Trajectories")
plt.xlim(-100, 100)
plt.xlabel("Position")
plt.ylabel("Time step")
plt.show()

# %% [markdown]

"""
On observe que les trajectoires sont variables. Elles sont essentiellement aléatoires mais on peut constater que la position de la particule est relativement symétrique autour de l'origine.

Essayons de reproduire la distribution de probabilité de la position de la particule après un certain nombre de pas de temps en échantillonnant un grand nombre de trajectoires.
"""

# %%
# | label: fig-random-walk-distribution
# | fig-cap: Distribution de probabilité de la position d'une particule effectuant une marche aléatoire simple en une dimension après x pas de temps
# | fig-alt: Histogramme des positions de la particule après x pas de temps.
num_trajectories = 10_000
num_steps = [500, 1_000, 2000]

plt.figure(figsize=(8, 6))

for steps in num_steps:
    positions = np.array([draw_trajectory(steps)[-1] for _ in range(num_trajectories)])
    print(f"After {steps} steps: Mean = {np.mean(positions)}, Variance = {np.var(positions)}")

    plt.hist(positions, bins=50, density=True, alpha=0.7, histtype='step', lw=2, label=f'{steps} steps')

plt.xlabel("Position")
plt.ylabel("Probability Density")
plt.legend()  
plt.show()


# %% [markdown]

"""
Il apparait très clairement que la distribution de probabilité de la position de la particule après un grand nombre de pas de temps ressemble à une distribution Gaussien centrée dont la variance augmente avec le nombre de pas effectués.

## Marche aléatoire asymétrique

Ici, nous avons modélisé le phénomène de diffusion via une marché aléatoire symétrique. Amusons-nous à considérer maintenant une marche aléatoire asymétrique, où la particule a une probabilité $p \neq 0.5$ de se déplacer vers la droite à chaque pas de temps. Prenons par exemple $p=0.52$.
"""

# %%
# | label: fig-random-walk-asymmetric-examples
# | fig-cap: Quelques trajectoires possibles d'une particule effectuant une marche aléatoire asymétrique en une dimension avec $p=0.52$.
# | fig-alt: Trajectoire de la marche aléatoire asymétrique.
num_trajectories = 5
num_steps = 1_000
p = 0.52

plt.figure(figsize=(8, 6))

for _ in range(num_trajectories):
    trajectory = draw_trajectory(num_steps, p=p)
    plt.plot(trajectory, range(num_steps), alpha=0.7)

plt.xlabel("Position")
plt.ylabel("Time step")
plt.show()


# %% [markdown]

"""
Il apparait très clairement que les trajectoires dérivent vers la droite, ce qui est intuitif. Il est possible de montrer que la marche aléatoire symétrique modélise un processus de diffusion pur, tandis que la marche aléatoire asymétrique modélise un processus de diffusion avec advection (un déplacement moyen non nul).

Observons cela en échantillonnant la distribution de probabilité de la position de la particule après un certain nombre de pas de temps dans le cas asymétrique.
"""

# %%
# | label: fig-random-walk-asymmetric-distribution
# | fig-cap: Distribution de probabilité de la position d'une particule effectuant une marche aléatoire asymétrique en une dimension après x pas de temps avec $p=0.52$.
# | fig-alt: Histogramme des positions de la particule après x pas de temps dans le cas asymétrique.
num_trajectories = 10_000
num_steps = [500, 1_000, 2000]
p = 0.52

plt.figure(figsize=(8, 6))

for steps in num_steps:
    positions = np.array([draw_trajectory(steps, p=p)[-1] for _ in range(num_trajectories)])
    print(f"After {steps} steps: Mean = {np.mean(positions)}, Variance = {np.var(positions)}")

    plt.hist(positions, bins=50, density=True, alpha=0.7, histtype='step', lw=2, label=f'{steps} steps')

plt.xlabel("Position")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

# %% [markdown]

"""
Ici, la moyenne de chaque distribution est non nulle et ce n'est pas une fluctuation statistique: le gaz de particule se déplace bel et bien vers la droite.

## Marche aléatoire avec dépendance temporelle

Dans les exemples donnés précédemment, chaque pas est tiré aléatoirement de façon indépendante du passé. Cela se voir clairement dans la fonction `draw_trajectory` où on tire chaque pas avant d'en déduire la trajectoire.

Dans cet exemple, nous proposons de complexifier le modèle en introduisant une dépendance avec le passé avec une marche renforcée. L'idée est que la probabilité de se déplacer à droite ou à gauche dépend du nombre de fois où la chaine s'est déplacée à droite ou à gauche. Autrement dit, le pas à l'instant $t$ dépend de l'historique des pas précédents.

Soit $D_n$ la variable aléatoire représentant la direction du déplacement au $n$-ième pas de temps.

$$
\mathbb{P}(D_n = +1) = \frac{1 + R_{n-1}}{2 + (n-1)}, \quad
$$

où $R_{n-1}$ est le nombre de déplacements vers la droite effectués jusqu'au pas $n-1$.
"""

# %%
# | echo: true
def draw_reinforced_trajectory(num_steps):
    trajectory = np.zeros(num_steps)
    right_moves = 0 #Counter for right moves

    for t in range(1, num_steps):
        p_right = ( 1 + right_moves ) / (2 + (t-1))
        step = np.random.choice([-1, 1], p=[1 - p_right, p_right])
        trajectory[t] = trajectory[t-1] + step

        if step == 1:
            right_moves += 1

    return trajectory

# %%
# | label: fig-reinforced-random-walk-trajectories
# | fig-cap: Quelques trajectoires possibles d'une particule effectuant une marche aléatoire renforcée en une dimension.
# | fig-alt: Trajectoires de marche aléatoire renforcée
num_trajectories = 5
num_steps = 1_000

plt.figure(figsize=(8, 6))

for _ in range(num_trajectories):
    trajectory = draw_reinforced_trajectory(num_steps)
    plt.plot(trajectory, range(num_steps), alpha=0.7)

plt.xlabel("Position")
plt.ylabel("Time step")
plt.show()

# %% [markdown]

"""
On observe que le comportement des trajectoires est complètement différent de celui des marches aléatoires considérées précédemment. En effet, la dépendance temporelle introduite dans le modèle crée une sorte d'effet de mémoire qui influence fortement la dynamique de la particule. Une particule qui commence à se déplacer vers la droite aura tendance à continuer dans cette direction, et vice versa pour la gauche. Cela conduit à des trajectoires plus "cohérentes" où la particule ne change pas fréquemment de direction, contrairement aux marches aléatoires indépendantes où les changements de direction sont fréquents et aléatoires.

Il est important de souligner que ce type de trajectoire provient d'un modèle. Il est pertinent à la fois de comprendre comment ces modèles se comportant, mais aussi de se demander dans quels contextes physiques réels ils peuvent s'appliquer. Par exemple, des phénomènes de marche aléatoire renforcée peuvent être observés dans certains systèmes biologiques où des agents (comme des cellules ou des animaux) modifient leur comportement en fonction de leurs expériences passées.
"""
