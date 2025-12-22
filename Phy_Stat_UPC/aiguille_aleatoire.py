# %% [markdown]
# ---
# title: "Aiguille de Buffon"
# description: "Simulation de l'expérience de l'aiguille de Buffon pour estimer la valeur de $\pi$."
# author: "Sacha Guerrini"
# date: today
# format: html
# jupyter: python3
# number-figures: true
# execute:
#   echo: false
# ---

# %%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from tqdm import tqdm

plt.style.use(
    "../matplotlib_config/paper.mplstyle"
)

sns.color_palette('husl')

# %% [markdown]

"""
Code source disponible [ici](https://github.com/sachaguerini/tutorials/tree/main/Phy_Stat_UPC/aiguille_aleatoire.py)

## Contexte

L'aiguille de Buffon est une expérience de probabilité classique qui consiste à laisser tomber une aiguille de longueur $\ell$ sur un plan avec des lignes parallèles espacées de distance $2a$.
Il est possible de déterminer la probabilité que l'aiguille croise une ligne en fonction de la longueur de l'aiguille et de l'espacement des lignes. Notons $P$ cette probabilité. On peut montrer que

$$
P = \frac{ 2 \ell}{\pi a}.
$$

Dans ce notebook, nous allons simuler cette expérience pour retrouver ce résultat analytique.

L'écriture de ce notebook est motivé par un exercice du cours de Physique Statistique 2024-2025 de l'Université Paris Cité que j'ai enseigné dans le cadre de ma mission d'enseignement doctorale.
"""

# %% [markdown]

"""
## Simulation de l'expérience

Commençons par écrire quelques fonctions pour simuler l'expérience de l'aiguille de Buffon.

Pour simuler le lancer d'une aiguille, nous avons besoin d'échantillonner la position de l'aiguille et son angle par rapport à un axe arbitraire. On prendra ici l'axe des abscisses.

Les fonctions `sample_x` et `sample_y` permettent d'échantillonner la position. La fonction `sample_angle` permet d'échantillonner l'angle. `sample_needle` combine ces trois fonctions pour simuler le lancer d'une aiguille. Voir script disponible [ici](https://github.com/sachaguerini/tutorials/tree/main/Phy_Stat_UPC/aiguille_aleatoire.py)
"""

# %%
def sample_y(num_samples):
    """
    Samples the position of the needle centre along the y-axis.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    
    Returns
    -------
    np.ndarray
        Array of shape (num_samples,) containing the x positions.
    """
    return np.random.uniform(0, 1, num_samples)

def sample_x(num_samples, a):
    """
    Samples the position of the needle centre along the x-axis.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    a : float
        Half the distance between the lines.
    
    Returns
    -------
    np.ndarray
        Array of shape (num_samples,) containing the y positions.
    """
    return np.random.uniform(-a, a, num_samples)

def sample_angle(num_samples):
    """
    Samples the angle of the needle with respect to the x-axis.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    
    Returns
    -------
    np.ndarray
        Array of shape (num_samples,) containing the angles in radians.
    """
    return np.random.uniform(0, 2*np.pi, num_samples)

def sample_needle(num_samples, a):
    """
    Samples the position and angle of the needle.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    a : float
        Half the distance between the lines.
    
    Returns
    -------
    tuple of np.ndarray
        Tuple containing three arrays of shape (num_samples,):
        - x positions
        - y positions
        - angles in radians
    """
    x = sample_x(num_samples, a)
    y = sample_y(num_samples)
    theta = sample_angle(num_samples)
    return x, y, theta

# %% [markdown]
"""
Notons que nous avons une certaine liberté dans les choix de modélisation du problème. Dans cette implémentation, nous supposerons la distribution suivante pour les variables aléatoires :
- $X$ (position selon l'axe des x) est uniformément distribuée entre $-a$ et $a$.
- $Y$ (position selon l'axe des y) est uniformément distribuée entre $0$ et $1$.
- $\theta$ (angle) est uniformément distribuée entre $0$ et $2\pi$.

Les trois variables sont indépendantes.

Les résultats ne doivent pas dépendre du choix de la distribution sur l'axe des $y$ tant qu'elle est indépendante des autres variables. Dans le calcul analytique, nous intégrerons la distribution de $Y$ et elle ne contribuera pas au résultat final. Le choix de l'intervalle $[0, 1]$ est arbitraire. Pour l'axe des x, nous devons nous assurer que l'aiguille peut traverser une ligne, d'où le choix de l'intervalle $[-a, a]$. Cela limite aussi l'étude à une seule ligne car les motifs se répètent périodiquement. Cela simplifie le problème en éliminant cette symétrie.

Ci-dessous, nous implémentons la fonction `plot_needle` pour visualiser un lancer d'aiguille.
"""

# %%
def plot_needle(ax, x, y, angle, l, color='r', label=None):
    """
    Plot a needle on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x : float
        The x position of the needle centre.
    y : float
        The y position of the needle centre.
    angle : float
        The angle of the needle in radians.
    l : float
        The length of the needle.
    """
    x_1 = x + l * np.cos(angle)
    y_1 = y + l * np.sin(angle)
    x_2 = x - l * np.cos(angle)
    y_2 = y - l * np.sin(angle)
    ax.plot([x_1, x_2], [y_1, y_2], color=color, label=label)
    ax.scatter([x], [y], color=color)

# %%
# |label: fig-needle
# |fig-cap: "Visualisation des aiguilles échantillonnées aléatoirement."
fig, axs = plt.subplots(1, 1, figsize=(8,8)) 

num_samples = 10
l = 0.8
a = 1.0

x, y, angle = sample_needle(num_samples, a)

for i in range(num_samples):
    plot_needle(axs, x[i], y[i], angle[i], l)

# Plot lines
axs.axvline(x=-a, color='black', linestyle='--')
axs.axvline(x=a, color='black', linestyle='--')
axs.axvline(x=0, color='black', linestyle='-')

axs.add_patch(
    Rectangle(
        (-a, 0), 2*a, 1,
        facecolor='lightgrey',
        alpha=0.3,
        edgecolor='none',
        zorder=0,
        label='Midpoint sampling region'
    ),
)

axs.text(-1.2, 1.8, r'$a=1.0$', fontsize=14, ha='center')
axs.text(-1.2, 1.7, r'$l=0.8$', fontsize=14, ha='center')

plt.xlim(-1.5*a, 1.5*a)
plt.ylim(-1, 2)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
axs.legend(
    bbox_to_anchor=(0.5, -0.1),
    frameon=False,
    fontsize=16
)
plt.show()

# %% [markdown]
"""
On peut maintenant implémenter des fonctions pour déterminer si une aiguille croise la ligne $x=0$. Nous pouvons adopter deux stratégies:
- Vérifier si l'aiguille croise en utilisant les extrémités de l'aiguille (`is_needle_crossing_endpoint`).
- Vérifier si l'aiguille croise en utilisant le centre (`is_needle_crossing_middle`).
"""

# %%
def is_needle_crossing_endpoint(x, angle, l):
    """
    Check if the needle crosses the line x=0 using the endpoint

    Parameters
    ----------
    x : np.ndarray
        Positions of the needles along the x-axis
    angle : np.ndarray
        Angle of the needles along the x-axis
    l : float
        Length of the needle
    """
    # Compute the endpoints
    x_1 = x + l * np.cos(angle)
    x_2 = x - l * np.cos(angle)
    return np.sign(x_1) != np.sign(x_2)

def is_needle_crossing_middle(x, angle, l):
    """
    Check if the needle crosses the line x=0 using the middle

    Parameters
    ----------
    x : np.ndarray
        Positions of the needles along the x-axis
    angle : np.ndarray
        Angle of the needles along the x-axis
    l : float
        Length of the needle
    """
    return (y >= -l*np.abs(np.cos(angle))) & (x <= l*np.abs(np.cos(angle)))

# %%
# | echo: True
np.all( is_needle_crossing_endpoint(x, angle, l) == is_needle_crossing_middle(x, angle, l))

# %%
# |label: fig-needle-crossing
# |fig-cap: "Visualisation des aiguilles échantillonnées aléatoirement avec indication de celles qui traversent la ligne (vert)."
fig, axs = plt.subplots(1, 1, figsize=(8, 8))

bool_crossing = is_needle_crossing_endpoint(x, angle, l)

for i in range(num_samples):
    color = 'g' if bool_crossing[i] else 'r'
    label = 'Crossing needle' if bool_crossing[i] else 'Non-crossing needle'
    plot_needle(axs, x[i], y[i], angle[i], l, color=color, label=label)

# Plot lines
axs.axvline(x=-a, color='black', linestyle='--')
axs.axvline(x=a, color='black', linestyle='--')
axs.axvline(x=0, color='black', linestyle='-')

axs.add_patch(
    Rectangle(
        (-a, 0), 2*a, 1,
        facecolor='lightgrey',
        alpha=0.3,
        edgecolor='none',
        zorder=0,
        label='Midpoint sampling region'
    )
)

axs.text(-1.2, 1.8, r'$a=1.0$', fontsize=14, ha='center')
axs.text(-1.2, 1.7, r'$l=0.8$', fontsize=14, ha='center')

plt.xlim(-1.5*a, 1.5*a)
plt.ylim(-1, 2)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
handles, labels = axs.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs.legend(
    by_label.values(),
    by_label.keys(),
    bbox_to_anchor=(1.0, -0.1),
    frameon=False,
    fontsize=16,
    ncol=2
)
plt.show()


# %% [markdown]

"""
## Calcul de la probabilité de croisement

On peut maintenant écrire une fonction pour estimer la probabilité de croisement en échantillonnant un grand nombre d'aiguilles et en comptant le nombre de croisements.
"""

# %%
# | echo: true
def compute_probability(num_samples, a, l):
    """
    Computes the probability of crossing the line x=0.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    a : float
        Half the distance between the lines.
    l : float
        Length of the needle.

    Returns
    -------
    float
        Estimated probability of crossing the line.
    """
    x, _, angle = sample_needle(num_samples, a)
    bool_crossing = is_needle_crossing_endpoint(x, angle, l)
    return np.mean(bool_crossing)

# %%
# | echo: true
num_samples = 100_000

proba = compute_probability(num_samples, a=a, l=l)
print(f"Estimated probability: {proba:.4f}")
print(f"Target probability: {2*l/(np.pi*a):.4f}")

# %% [markdown]

"""
Les deux valeurs sont en bon accord, ce qui valide notre simulation. Nous pouvons maintenant étudier la convergence de l'estimation de la probabilité en fonction du nombre d'échantillons. En particulier, en répétant l'expérience plusieurs fois pour un nombre fixe d'échantillons, nous pouvons estimer la variance de l'estimation ce qui permettra d'estimer un intervalle de confiance pour la valeur de la probabilité.

Ecrivons d'abord la fonction `sample_proba` qui répète l'expérience plusieurs fois pour un nombre fixe d'échantillons.
"""

# %%
# | echo: true
def sample_proba(num_iter, num_samples, a, l):
    """
    Samples the probability of crossing the line x=0 multiple times.

    Parameters
    ----------
    num_iter : int
        Number of iterations to perform.
    num_samples : int
        Number of samples to generate for each iteration.
    a : float
        Half the distance between the lines.
    l : float
        Length of the needle.

    Returns
    -------
    np.ndarray
        Array of shape (num_iter,) containing the estimated probabilities.
    """
    return np.array(
        [
            compute_probability(num_samples, a, l)
            for _ in range(num_iter) # Using a for loop is not optimal here
        ]
    )

# %% [markdown]

"""
On peut maintenant répéter l'expérience pour différents nombres d'échantillons et tracer la convergence de l'estimation de la probabilité.
"""

# %%
num_samples_list =[10, 100, 1_000, 10_000, 100_000, 1_000_000]

num_iter = 1000
proba_list = [
    sample_proba(num_iter, num_samples, a, l) for num_samples in tqdm(num_samples_list)
]

# %%
# |label: fig-proba-convergence
# |fig-cap: "Convergence de l'estimation de la probabilité de croisement en fonction du nombre d'échantillons."
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

mean_proba = [np.mean(probas) for probas in proba_list]
std_proba = [np.std(probas) for probas in proba_list]

ax.errorbar(
    num_samples_list,
    mean_proba,
    yerr=std_proba,
    fmt='o',
    label='Estimated probability',
    capsize=5
)

ax.axhline(
    2*l/(np.pi*a),
    color='k',
    linestyle='--',
    label='Analytical probability'
)

plt.xlabel('Number of samples')
plt.ylabel('Probability of crossing')
plt.legend(fontsize=16)
plt.xscale('log')

plt.show()

# %% [markdown]

"""
On observe deux choses:
1. La moyenne de l'estimation de la probabilité converge vers la valeur analytique lorsque le nombre d'échantillons augmente.
2. L'écart-type de l'estimation diminue avec le nombre d'échantillons, ce qui est attendu car l'erreur d'estimation diminue avec le nombre d'échantillons.

Pouvons-nous quantifier cette diminution de l'erreur d'estimation? Pour cela, traçons l'écart-type en fonction du nombre d'échantillons sur une échelle log-log.
"""

# %%
# |label: fig-proba-std
# |fig-cap: "Écart-type de l'estimation de la probabilité en fonction du nombre d'échantillons."
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(
    num_samples_list,
    std_proba,
    'o-',
    label='Estimated standard deviation'
)

ax.set_xscale('log')
ax.set_yscale('log')

plt.xlabel('Number of samples')
plt.ylabel('Standard deviation of probability estimate')
plt.legend(fontsize=16)
plt.grid()
plt.show()

# %% [markdown]

"""
On observe une relation de type loi de puissance entre l'écart-type et le nombre d'échantillons. Pour quantifier cette relation, nous pouvons effectuer une régression linéaire sur les données log-log.
"""

# %%
# | echo: true
log_num_samples = np.log(num_samples_list)
log_std_proba = np.log(std_proba)
slope, intercept, r_value, p_value, std_err = stats.linregress(log_num_samples, log_std_proba)

print(f"Slope: {slope:.4f} ± {std_err:.4f}")

# %% [markdown]

"""
On observe que la variance de l'esimateur décroît en $\propto N^{-1/2}$. Connaissez-vous un théorème important en statistique qui explique ce comportement?

C'est le théorème central limite, qui stipule que la moyenne d'un grand nombre de variables aléatoires indépendantes et identiquement distribuées suit une distribution normale dont l'écart-type diminue en $\propto N^{-1/2}$. Essayons de le visualiser!
"""

# %%
# |label: fig-proba-hist
# |fig-cap: "Histogrammes des estimations de la probabilité comparés à une distribution normale."
plt.figure(figsize=(8, 8))

mu = np.mean(proba_list[-1])
std = np.std(proba_list[-1])
x = np.linspace(mu - 4*std, mu + 4*std, 1000)

y = stats.norm.pdf(x, mu, std)

plt.plot(x, y, 'k--', label='Normal distribution')
plt.axvline(x=mu, color='r', linestyle='-', label='Empirical estimate of the probability')
plt.axvline(x=2*l/(np.pi*a), color='g', linestyle='--', label='Analytical probability')

plt.hist(
    proba_list[-1],
    bins=100,
    density=True,
    alpha=0.6,
    label='Histogram of the predicted probabilities'
)

plt.xlabel('Probability of crossing')
plt.ylabel('Density')
plt.legend(fontsize=10)
plt.show()

# %% [markdown]

"""
On peut même regarder comments l'histogramme des probabilités évolue avec le nombre d'échantillons.
"""

# %%
# |label: fig-proba-hist-evolution
# |fig-cap: "Évolution de l'histogramme des estimations de la probabilité avec le nombre d'échantillons."
plt.figure(figsize=(8, 8))

for i, num_samples in enumerate(num_samples_list):
    if num_samples > 1000:
        plt.hist(
            proba_list[i],
            bins=100,
            density=True,
            alpha=0.4,
            range=(0.49, 0.53),
            label=f'Num samples: {num_samples}'
        )

plt.axvline(x=2*l/(np.pi*a), color='r', linestyle='--', label='Analytical probability')

plt.xlabel('Probability of crossing')
plt.ylabel('Density')
plt.legend(fontsize=12)
plt.show()

# %% [markdown]

"""
## Estimation de la valeur de $\pi$

On peut aussi inverser la relation analytique pour estimer la valeur de $\pi$ à partir de l'estimation de la probabilité de croisement. En particulier, il est intéressant de propager l'incertitude sur l'estimation de la probabilité vers une incertitude sur l'estimation de $\pi$. En effet on a

$$
\pi = \frac{2 \ell}{a P}.
$$

En supposant que la distance $a$ et la longueur $\ell$ sont connues exactement, l'incertitude sur $\pi$ est donnée par

$$
\Delta \pi = \left| \frac{\partial \pi}{\partial P} \right| \Delta P = \frac{2 \ell}{a P^2} \Delta P.
$$

Plottons ce résultat.
"""

# %%
# |label: fig-pi-estimate
# |fig-cap: "Estimation de la valeur de $\pi$ en fonction du nombre d'échantillons."
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

mean_pi = [2*l/(a*mean_p) for mean_p in mean_proba]
std_pi = [ (2*l)/(a*(mean_p**2)) * std_p for mean_p, std_p in zip(mean_proba, std_proba)]

ax.errorbar(
    num_samples_list,
    mean_pi,
    yerr=std_pi,
    fmt='o',
    label=r'Estimated value of $\pi$',
    capsize=5
)
ax.axhline(
    np.pi,
    color='k',
    linestyle='--',
    label=r'True value of $\pi$'
)

plt.xlabel('Number of samples')
plt.ylabel(r'Estimated value of $\pi$')
plt.legend(fontsize=16)
plt.xscale('log')

plt.show()

# %% [markdown]

"""
Cela fonctionne bien! 🥳 On peut calculer l'estimation avec le plus grand nombre d'échantillons
"""

# %%
# | echo: true
mean_pi[-1], std_pi[-1]
print(f"Estimated value of pi: {mean_pi[-1]:.6f} ± {std_pi[-1]:.6f}")

# %% [markdown]

"""
On retrouve la valeur de $\pi$ à $10^{-3}$ près avec $10^6$ échantillons! Théoriquement, pour atteindre une précision de $10^{-n}$, il faudrait environ $10^{2n}$ échantillons car l'erreur diminue en $\propto N^{-1/2}$. Mais cela devient très coûteux en temps de calcul pour des précisions élevées...
"""

# %% [markdown]

"""
On observe qu'à mesure que le nombre d'échantillons augmente, l'histogramme des probabilités estimées devient plus concentré autour de la valeur analytique, illustrant la convergence de l'estimateur et le théorème central limite en action.

## Conclusion

Dans ce notebook, nous avons simulé l'expérience de l'aiguille de Buffon pour estimer la probabilité de croisement d'une aiguille avec des lignes parallèles. Nous avons validé notre simulation en comparant les résultats numériques avec la solution analytique connue. Nous avons également étudié la convergence de l'estimation de la probabilité en fonction du nombre d'échantillons, en observant que l'écart-type de l'estimation diminue conformément au théorème central limite.
"""