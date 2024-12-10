import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from concurrent.futures import ProcessPoolExecutor
from MKP_problems import MKP1, MKP2, MKP3, MKP4, MKP5, MKP6, MKP7, MKP8, MKP9, MKP10
from BPSO1 import BPSO1
from GA1 import GA1
from BPSO2 import BPSO2
from GA2 import GA2
from BPSO3 import BPSO3
from GA3 import GA3

# Global parameters
D = 28
N = 30
Tmax = 1000
step = 25
test_runs = 30

def run_single_algorithm(algorithm, func, N, D, Tmax, step, run_id):
    """Runs a single instance of an algorithm."""
    print(f"Run {run_id} - {algorithm.__name__}")
    try:
        result = algorithm(func, N=N, D=D, Tmax=Tmax, step=step)
        return result
    except Exception as e:
        print(f"Error during the execution of {algorithm.__name__} in run {run_id}: {e}")
        return []

def analyze_final_results(file_name, problem_name):
    final_values = {
        "BPSO1": [],
        "BPSO2": [],
        "BPSO3": [],
        "GA1": [],
        "GA2": [],
        "GA3": []
    }

    try:
        with open(file_name, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found!")
        return

    section_indices = {}
    for key in final_values.keys():
        try:
            section_indices[key] = lines.index(f"{key} Results:\n") + 1
        except ValueError:
            section_indices[key] = None

    section_keys = list(section_indices.keys())
    for i, key in enumerate(section_keys):
        start = section_indices[key]
        if start is None:
            continue
        end = section_indices[section_keys[i + 1]] - 1 if i + 1 < len(section_keys) and section_indices[section_keys[i + 1]] is not None else len(lines)
        for line in lines[start:end]:
            if line.strip() and not line.startswith(f"{key} Results:"):
                try:
                    final_values[key].append(float(line.strip().split()[-1]))
                except ValueError:
                    pass

    stats = {}
    for key, values in final_values.items():
        if values:
            stats[key] = {
                "Best": np.max(values),
                "Mean": np.mean(values),
                "StdDev": np.std(values)
            }
        else:
            stats[key] = {
                "Best": None,
                "Mean": None,
                "StdDev": None
            }

    csv_filename = f"./Over_all_results/{problem_name}_overall_results_parallel.csv"
    try:
        with open(csv_filename, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Algorithm", "Best", "Mean", "StdDev"])
            for key, stat in stats.items():
                csvwriter.writerow([key, stat["Best"], stat["Mean"], stat["StdDev"]])
        print(f"Results saved to '{csv_filename}'.")
    except IOError as e:
        print(f"Error saving results: {e}")

def run_algorithms(problem_name):
    global D
    if problem_name in ["MKP1", "MKP2", "MKP3", "MKP4", "MKP5", "MKP6"]:
        D = 28
        func = eval(problem_name)
    elif problem_name in ["MKP7", "MKP8"]:
        D = 105
        func = eval(problem_name)
    elif problem_name in ["MKP9", "MKP10"]:
        D = 60
        func = eval(problem_name)
    else:
        print("Ce nom n'est pas dans la liste !")
        return

    algorithms = [BPSO1, GA1, BPSO2, GA2, BPSO3, GA3]
    results = {alg.__name__: [] for alg in algorithms}

    with ProcessPoolExecutor() as executor:
        for algorithm in algorithms:
            futures = [
                executor.submit(run_single_algorithm, algorithm, func, N, D, Tmax, step, run_id)
                for run_id in range(1, test_runs + 1)
            ]
            for future in futures:
                results[algorithm.__name__].append(future.result())

    results_filename = f"./Results/results{problem_name}.txt"
    with open(results_filename, "w") as f:
        f.write(f"{problem_name} Results:\n")
        for algorithm in results:
            f.write(f"{algorithm} Results:\n")
            for run in results[algorithm]:
                f.write(" ".join(map(str, run)) + "\n")
            f.write("\n")

    print("Detailed results saved to results.txt")

    iterations = np.arange(step, Tmax + step, step)
    plt.figure(figsize=(12, 7))

    for algorithm in results:
        avg_scores = np.mean(results[algorithm], axis=0)
        plt.plot(iterations, avg_scores, label=algorithm, marker='o')

    plt.xlabel("Iterations")
    plt.ylabel("LOG Coût Moyens Meilleurs")
    plt.title(f"Comparaison des Méthodes - {problem_name}")
    plt.legend()
    plt.grid()
    plt.yscale('log')

    plot_filename = f"./Graphe/{problem_name}_comparison_graph.png"
    plt.savefig(plot_filename)
    plt.show()

    print(f"Graph saved to {plot_filename}")

    analyze_final_results(results_filename, problem_name)

if __name__ == "__main__":
    start_time = time.time()
    problem_name = input("Donner le nom de problème (ex: MKP1, MKP2, ...): ")
    run_algorithms(problem_name)
    print(f"Temps d'exécution total : {time.time() - start_time:.2f} seconds")