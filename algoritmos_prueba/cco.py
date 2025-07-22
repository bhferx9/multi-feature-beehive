import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# --- Función objetivo para clustering ---
def clustering_fitness(solution, data, k):
    centroids = solution.reshape((k, data.shape[1]))
    # Asignación de clusters
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    # Suma de distancias intra-cluster
    total_dist = sum(np.linalg.norm(data[i] - centroids[labels[i]]) for i in range(len(data)))
    return total_dist

# --- Mapa logístico (caótico) ---
def logistic_map(x, r=3.99):
    return r * x * (1 - x)

# --- Cluster Chaotic Optimization (CCO) ---
class CCO:
    def __init__(self, data, k, bounds, num_nests=25, max_iter=100, pa=0.25, step_size=0.5):
        self.data = data
        self.k = k
        self.dim = k * data.shape[1]
        self.bounds = bounds
        self.num_nests = num_nests
        self.max_iter = max_iter
        self.pa = pa
        self.step_size = step_size

        # Inicializar nidos (posiciones de centroides)
        self.nests = np.random.uniform(bounds[0], bounds[1], (num_nests, self.dim))
        self.fitness = np.array([clustering_fitness(n, data, k) for n in self.nests])

        self.best_idx = np.argmin(self.fitness)
        self.best_nest = self.nests[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        self.best_values = []

        # Inicializar variable caótica
        self.chaotic_var = np.random.rand()

    def optimize(self):
        for _ in range(self.max_iter):
            new_nests = self.nests.copy()

            # Caos + paso
            for i in range(self.num_nests):
                # Actualizar variable caótica
                self.chaotic_var = logistic_map(self.chaotic_var)
                step = self.step_size * (np.random.rand(self.dim) - 0.5) * 2 * self.chaotic_var
                new_nests[i] += step
                new_nests[i] = np.clip(new_nests[i], self.bounds[0], self.bounds[1])

            # Evaluar nuevos nidos
            new_fitness = np.array([clustering_fitness(n, self.data, self.k) for n in new_nests])

            # Reemplazar si mejora
            improved = new_fitness < self.fitness
            self.nests[improved] = new_nests[improved]
            self.fitness[improved] = new_fitness[improved]

            # Abandonar algunos nidos aleatorios
            abandon = np.random.rand(self.num_nests) < self.pa
            num_abandon = np.sum(abandon)
            if num_abandon > 0:
                self.nests[abandon] = np.random.uniform(self.bounds[0], self.bounds[1], (num_abandon, self.dim))
                self.fitness[abandon] = np.array(
                    [clustering_fitness(n, self.data, self.k) for n in self.nests[abandon]])

            # Actualizar mejor solución
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_nest = self.nests[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]

            self.best_values.append(self.best_fitness)

        return self.best_nest.reshape((self.k, self.data.shape[1])), self.best_fitness

    def plot_convergence(self):
        plt.plot(self.best_values)
        plt.title('Convergencia del CCO')
        plt.xlabel('Iteración')
        plt.ylabel('Fitness (suma de distancias)')
        plt.grid(True)
        plt.show()

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Generar datos
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    bounds = [np.min(X), np.max(X)]

    # Ejecutar CCO
    cco = CCO(data=X, k=3, bounds=bounds, max_iter=100)
    best_centroids, best_fit = cco.optimize()

    # Mostrar resultados
    print("Mejores centroides encontrados:\n", best_centroids)
    cco.plot_convergence()

    # Visualización de clusters
    distances = np.linalg.norm(X[:, np.newaxis] - best_centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    plt.scatter(best_centroids[:, 0], best_centroids[:, 1], c='red', marker='X', s=200)
    plt.title("Clusters encontrados por CCO")
    plt.grid(True)
    plt.show()
