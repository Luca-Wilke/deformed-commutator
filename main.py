import numpy as np
from matplotlib import pyplot as plt
import random as rand

def q_def_comm(q: int, A: np.array, B: np.array) -> np.array:
    return np.matmul(A, B) - (q * np.matmul(B, A)) 

def gen_traceless(n: int, val_min: float, val_max: float) -> np.array:
    entries = [[0 if j==i else (rand.uniform(val_min, val_max)) for j in range(n)] for i in range(n)]
    return np.array(entries)

def gen_matrix(n: int, val_min: float, val_max: float) -> np.array:
    entries = [[(rand.uniform(val_min, val_max)) for j in range(n)] for i in range(n)]
    return np.array(entries)

def frob(A: np.array) -> float:
    return np.linalg.norm(A, 'fro')

def frob_squared(A: np.array) -> float:
    return np.square(frob(A))

def plot(A: np.array, B: np.array, q_max: int):
    data_f = []
    data_g = []
    data_diff = []
    data_quotient = []
    for q in range(q_max):
        val_f = frob_squared(q_def_comm(q / 25, A, B))
        val_g = (1 + np.square(q / 25)) * frob_squared(A) * frob_squared(B)
        data_f.append(val_f)
        data_g.append(val_g)
        data_diff.append(val_g - val_f)
        data_quotient.append(-1 if val_f == 0 else (val_g / val_f))
    print(data_quotient)
    plt.title("n=2")
    plt.xlabel("q")
    plt.ylabel("y")
    plt.plot([i / 25 for i in range(q_max)], data_f, 'r')
    plt.plot([i / 25 for i in range(q_max)], data_g, 'g')
    # plt.plot([i / 25 for i in range(q_max)], data_diff, 'b')
    plt.plot([i / 25 for i in range(q_max)], data_quotient, 'y')
    plt.plot([i / 25 for i in range(q_max)], [1 for i in range(q_max)])
    # plt.plot([i / 25 for i in range(q_max)], [0 for i in range(q_max)])
    plt.show()

plot(gen_traceless(2, 0.0, 1.0), gen_matrix(2, 0.0, 1.0), 150)