from tkinter import filedialog
from tkinter import *
import numpy as np


def main():
    r = Tk()
    r.withdraw()
    r.fileName = filedialog.askopenfilename()
    path = r.fileName
    vectors = load_data(path)
    print(vectors)
    process_data(vectors)


def process_data(eye_vectors: np.ndarray):
    root = Tk()
    root.withdraw()
    root.fileName = filedialog.askopenfilename()
    path = root.fileName

    x, y = load_data(path)
    data = sort_data(x, y, eye_vectors)


def load_data(path: str) -> (np.ndarray, np.ndarray):
    data = np.loadtxt(path, delimiter=';', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def sort_data(x_pos: np.ndarray, y_pos: np.ndarray, eye_vectors: np.ndarray) -> list:
    sorted_vectors = list()
    vectors_x = list()
    vectors_y = list()
    x_values = np.unique(x_pos)
    y_values = np.unique(y_pos)
    for x in x_values:
        indices_x = np.where(x_pos == x)[0]
        for y in y_values:
            indices_y = np.where(y_pos == y)[0]
            indices = np.intersect1d(indices_x, indices_y)
            for i in indices:
                vectors_x.append(eye_vectors[0][i])
                vectors_y.append(eye_vectors[1][i])
            sorted_vectors.append((vectors_x, vectors_y))
    return sorted_vectors


if __name__ == "__main__":
    main()
