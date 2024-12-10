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
    return algorithm(func, N=N, D=D, Tmax=Tmax, step=step)


def run_algorithms(problem_name):
    global D
    if problem_name in ["MKP1", "MKP2", "MKP3", "MKP4", "MKP5", "MKP6"]:
        D = 28
        func = eval(problem_name)
    elif problem_name in ["MKP7", "MKP8"]:  # MKP8 requires D=105
        D = 105
        func = eval(problem_name)
    elif problem_name in ["MKP9", "MKP10"]:
        D = 60
        func = eval(problem_name)
    else:
        print("Ce nom n'est pas dans la liste ! ")
        return

    # List of algorithms to run
    algorithms = [BPSO1, GA1, BPSO2, GA2, BPSO3, GA3]

    # Container for results
    results = {alg.__name__: [] for alg in algorithms}

    # Parallel execution
    with ProcessPoolExecutor() as executor:
        for algorithm in algorithms:
            futures = [
                executor.submit(run_single_algorithm, algorithm, func, N, D, Tmax, step, run_id)
                for run_id in range(1, test_runs + 1)
            ]

            # Collect results as they complete
            for future in futures:
                results[algorithm.__name__].append(future.result())

    # Calculate overall statistics
    def calculate_overall_stats(results):
        overall_best = np.min(results)         # Overall best score across all runs
        overall_avg = np.mean(results)         # Overall average across all runs
        overall_std = np.std(results)          # Overall standard deviation across all runs
        return overall_best, overall_avg, overall_std

    overall_stats = {
        algorithm: calculate_overall_stats(results[algorithm])
        for algorithm in results
    }

    # Save only overall values to CSV
    csv_filename = f"./Over_all_results/{problem_name}_overall_results_parallel.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write headers
        csvwriter.writerow(["Algorithm", "Overall Best Score", "Overall Average", "Overall Standard Deviation"])
        # Write overall statistics
        for algorithm, stats in overall_stats.items():
            csvwriter.writerow([algorithm, *stats])

    print(f"Overall results saved to {csv_filename}")

    # Write detailed results to a text file
    with open(f"./Results/results{problem_name}.txt", "w") as f:
        f.write(f"{problem_name} Results:\n")
        for algorithm in results:
            f.write(f"{algorithm} Results:\n")
            for run in results[algorithm]:
                f.write(" ".join(map(str, run)) + "\n")
            f.write("\n")

    print("Detailed results saved to results.txt")

    # Plotting the results (using average scores for the plot)
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

    # Save plot with the MKP problem name
    plot_filename = f"./Graphe/{problem_name}_comparison_graph.png"
    plt.savefig(plot_filename)
    plt.show()

    print(f"Graph saved to {plot_filename}")


if __name__ == "__main__":
    start_time = time.time()
    problem_name = input("Donner le nom de problème (ex: MKP1, MKP2, ...): ")
    run_algorithms(problem_name)
    print(f"Temps d'exécution total : {time.time() - start_time:.2f} secondes")
