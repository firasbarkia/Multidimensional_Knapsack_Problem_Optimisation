import time
import numpy as np
import matplotlib.pyplot as plt
from MKP_problems import MKP1, MKP2, MKP3, MKP4, MKP5, MKP6, MKP7, MKP8, MKP9, MKP10
from BPSO1 import BPSO1
from GA1 import GA1
from BPSO2 import BPSO2
from GA2 import GA2

# Global parameters
D = 28
N = 30
Tmax = 1000
step = 25
test_runs = 30

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

    pso1_results = []
    ga_results = []
    tbpso_results = []
    ga2_results = []

    for run in range(test_runs):
        print(f"Run {run + 1} - BPSO1")
        pso1_results.append(BPSO1(func, N=N, D=D, Tmax=Tmax, step=step))
        print(f"Run {run + 1} - GA")
        ga_results.append(GA1(func, N=N, D=D, Tmax=Tmax, step=step))
        print(f"Run {run + 1} - TBPSO")
        tbpso_results.append(BPSO2(func, N=N, D=D, Tmax=Tmax, step=step))
        print(f"Run {run + 1} - GA2")
        ga2_results.append(GA2(func, N=N, D=D, Tmax=Tmax, step=step))

    # Calculate means
    pso1_means = np.mean(pso1_results, axis=0)
    ga_means = np.mean(ga_results, axis=0)
    tbpso_means = np.mean(tbpso_results, axis=0)
    ga2_means = np.mean(ga2_results, axis=0)

    iterations = np.arange(step, Tmax + step, step)

    # Save results
    with open("results.txt", "w") as f:
        f.write("BPSO1 Results:\n")
        for run in pso1_results:
            f.write(" ".join(map(str, run)) + "\n")
        f.write("\nGA Results:\n")
        for run in ga_results:
            f.write(" ".join(map(str, run)) + "\n")
        f.write("\nTBPSO Results:\n")
        for run in tbpso_results:
            f.write(" ".join(map(str, run)) + "\n")
        f.write("\nGA2 Results:\n")
        for run in ga2_results:
            f.write(" ".join(map(str, run)) + "\n")

    # Plotting the results
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, pso1_means, label="BPSO1", marker='o')
    plt.plot(iterations, ga_means, label="GA", marker='^')
    plt.plot(iterations, tbpso_means, label="TBPSO", marker='+')
    plt.plot(iterations, ga2_means, label="GA2", marker='v')
    plt.xlabel("Iterations")
    plt.ylabel("LOG Coût Moyens Meilleurs")
    plt.title("Comparaison des Méthodes")
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.savefig("comparison_graph.png")
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    problem_name = input("Donner le nom de problème (ex: MKP1, MKP2, ...): ")
    run_algorithms(problem_name)
    print(f"Temps d'exécution total : {time.time() - start_time:.2f} secondes")
