import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import os
import math
import subprocess

env = os.environ.copy()
env["OMP_PROC_BIND"] = "true"
env["OMP_PLACES"] = "cores"

MAX_CPUS = int(subprocess.Popen("./getcpus", stdout=subprocess.PIPE).stdout.read())+1

def strong_scaling():
    x = []
    y = []

    # run
    for cpus in range(1, MAX_CPUS):
        env["OMP_NUM_THREADS"] = str(cpus)
        tmp = subprocess.Popen(
            ["build/src/top.matrix_product", "1000", "1000", "1000"], 
            stdout=subprocess.PIPE,
            env=env).stdout.read().decode().strip().split("\t")
        
        x.append(int(tmp[3]))
        y.append(float(tmp[4]))

    # calc
    initial_time = y[0]
    y = initial_time / np.array(y)

    # plot
    plt.plot(x, y, label='Speedup')
    plt.plot(np.arange(1, x[-1], 0.1), np.arange(1, x[-1], 0.1), color="gray", linestyle='dashed', label="Ideal speedup")
    plt.grid(True, which="both", ls="--")
    plt.xscale("log", base=2)
    plt.ylim(0)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.xlabel("Number of cores")
    plt.ylabel("Speedup")
    plt.legend()
    plt.title("GEMM strong scaling")
    plt.tight_layout()
    plt.savefig("strong_scaling.png")

def weak_scaling():
    x = []
    y = []

    op_per_cpu = 1000 * 1000 * 1000

    # runs
    for cpus in range(1, MAX_CPUS):
        # that shoud give a ~constant amount of operations per core
        load = math.pow(op_per_cpu * cpus, 1/3)
        env["OMP_NUM_THREADS"] = str(cpus)
        tmp = subprocess.Popen(
            ["build/src/top.matrix_product", f"{load}", f"{load}", f"{load}"], 
            stdout=subprocess.PIPE,
            env=env).stdout.read().decode().strip().split("\t")

        x.append(int(tmp[3]))
        y.append(float(tmp[4]))
    
    # calc
    initial_time = y[0]
    y = initial_time / np.array(y)

    # plot
    plt.plot(x, y, label='Efficiency')
    plt.hlines([1], 1, x[-1], color="gray", linestyle="dashed", label="Ideal efficiency")
    plt.grid(True, which="both", ls="--")
    plt.xscale("log", base=2)
    plt.ylim(0)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.xlabel("Number of cores")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.title("GEMM weak scaling")
    plt.tight_layout()
    plt.savefig("weak_scaling.png")

if __name__ == "__main__":
    strong_scaling()
    weak_scaling()