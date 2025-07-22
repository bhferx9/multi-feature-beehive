import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import random

# Cargar datos
df = pd.read_csv("./extraccion_caracteristicas/resultados/dataset_caracteristicas_limpio.csv")
X = df.drop(columns=["archivo", "etiqueta"])
y = LabelEncoder().fit_transform(df["etiqueta"])

# Evaluación de precisión
def fitness(solution):
    if np.count_nonzero(solution) == 0:
        return 1.0
    X_sel = X.iloc[:, solution == 1]
    acc = cross_val_score(SVC(kernel='linear'), X_sel, y, cv=5).mean()
    return 1 - acc

# Mapa logístico (caótico)
def logistic_map(x, r=3.9):
    return r * x * (1 - x)

# Explorar usando dinámica caótica
def chaotic_exploration(solution, chaos_value):
    new_sol = solution.copy()
    for i in range(len(new_sol)):
        if logistic_map(chaos_value) > 0.6:
            new_sol[i] = 1 - new_sol[i]
    return new_sol

# Inicialización
n_nests = 20
n_features = X.shape[1]
n_iterations = 50
k_clusters = 3
chaos_init = random.random()

population = np.random.randint(0, 2, (n_nests, n_features))
fitness_values = np.array([fitness(sol) for sol in population])

best_idx = np.argmin(fitness_values)
best_solution = population[best_idx].copy()
best_fitness = fitness_values[best_idx]

# Algoritmo principal CCO
for iter in range(n_iterations):
    # Agrupar soluciones en clústeres
    kmeans = KMeans(n_clusters=k_clusters, n_init=10)
    labels = kmeans.fit_predict(population)

    for c in range(k_clusters):
        indices = np.where(labels == c)[0]
        if len(indices) == 0:
            continue
        avg_fitness = np.mean(fitness_values[indices])

        for i in indices:
            chaos_value = logistic_map(chaos_init)
            new_sol = chaotic_exploration(population[i], chaos_value)
            new_fit = fitness(new_sol)

            if new_fit < fitness_values[i]:
                population[i] = new_sol
                fitness_values[i] = new_fit
                if new_fit < best_fitness:
                    best_solution = new_sol.copy()
                    best_fitness = new_fit

    # Reemplazar peores soluciones
    pa = 0.25
    n_replace = int(pa * n_nests)
    worst_idx = fitness_values.argsort()[-n_replace:]
    for i in worst_idx:
        population[i] = np.random.randint(0, 2, n_features)
        fitness_values[i] = fitness(population[i])

# Resultados
selected_indices = np.where(best_solution == 1)[0]
selected_names = X.columns[selected_indices]

print("\nCaracterísticas seleccionadas:", selected_names.tolist())
print("Número de características:", len(selected_indices))
print("Precisión final con CCO:", 1 - best_fitness)

df_original = pd.read_csv('./extraccion_caracteristicas/resultados/dataset_caracteristicas_limpio.csv')
df_reducido = df_original[['archivo', 'etiqueta'] + list(selected_names)]
df_reducido.to_csv("./reduccion_caracteristicas/resultados/new_dataset_cco_reducido.csv", index=False)
print("✅ Dataset reducido con CCO guardado como: dataset_cco_reducido.csv")

