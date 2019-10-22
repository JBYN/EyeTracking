from tkinter import filedialog
from tkinter import *
import numpy as np
import os.path
import pandas as pd


dataframe = None


def main():
    r = Tk()
    r.withdraw()
    r.fileName = filedialog.askopenfilename()
    path = r.fileName
    vectors = load_data(path)
    process_data(vectors)


def process_data(eye_vectors: tuple, path1: str, filename: str, eye: str, method: str, mode: str, video: int,
                 duration: int) -> (list, list):
    # root = Tk()
    # root.withdraw()
    # root.fileName = filedialog.askopenfilename()
    # path = root.fileName
    print('start')
    path = path1 + filename
    x, y = load_data(path)
    # eye_vectors = load_data(eye_vectors_path)
    # print(eye_vectors_path)
    # print(eye_vectors[0].__len__())
    data = sort_data(x, y, eye_vectors)
    dist = calculate_dist(data)
    dist_filtered = remove_outliers_dist(dist)
    data_filtered = remove_outliers(data)
    # print(data_filtered)
    index = 0
    region_boundaries = list()
    variances = list()
    means = list()
    for i in data_filtered:
        index += 1
        mean_x = np.mean(i[0])
        mean_y = np.mean(i[1])
        mean_d = np.mean(dist_filtered[index - 1])
        std_x = np.std(i[0])
        std_y = np.std(i[1])
        std_d = np.std(dist_filtered[index - 1])
        variances.append([std_x/mean_x, std_y/mean_y, std_d/mean_d])
        means.append([mean_x, mean_y, mean_d])
        region_boundaries.append((mean_x - std_x, mean_y - std_y, mean_x + std_x, mean_y + std_y))

    header_cal = [['Video', 'Method', 'Type', 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8,
                  9, 9, 9],
                  ['', '', '', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis',
                  'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis']]
    header_test = [['Video', 'Method', 'Type', 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8,
                   9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16],
                   ['', '', '', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis',
                   'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis',
                    'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis', 'X', 'Y', 'Dis']]

    results_path = "C:/Users/Jo/Documents/School/Masterproef/Results_pupil_detection/"
    variances_filename = results_path + mode + "_" + eye + ".csv"
    if os.path.isfile(variances_filename) is False:
        global dataframe
        if dataframe is None:
            if mode == "Calibration":
                print("create")
                df = pd.DataFrame(columns=header_cal)
                df.set_index(['Video', 'Method', 'Type'], inplace=True)
                dataframe = df
            elif mode == "Test":
                df = pd.DataFrame(columns=header_test)
                df.set_index(['Video', 'Method', 'Type'], inplace=True)
                dataframe = df
    elif method == 'METHOD1':
        print("load")
        df = pd.read_csv(variances_filename,  skiprows=2, header=None, index_col=0)
        df.columns = header_cal
        df.set_index(["Video", "Method", "Type"], inplace=True)
        print(df)
        dataframe = df

    index = 1
    for j in variances:
        dataframe.at[(video, method, 'Var'), index] = j
        dataframe.at[(video, method, 'Mean'), index] = means[index - 1]
        index += 1

    if method == 'METHOD3':
        dataframe.reset_index(inplace=True, col_fill="")
        print(dataframe)
        dataframe.to_csv(variances_filename)
        dataframe = None

    # write duration to file
    headers = ['Video', 'METHOD1', 'METHOD2', 'METHOD3']
    duration_file = results_path + "Duration.csv"
    if os.path.isfile(duration_file) is False:
        df = pd.DataFrame(columns=headers)
        df.set_index('Video', inplace=True)
        df.at[video, method] = duration
        df.reset_index(inplace=True)
        df.to_csv(duration_file)
    else:
        print(method)
        df = pd.read_csv(duration_file, header=0, index_col="Video")
        df.at[video, method] = duration
        df.reset_index(inplace=True)
        df.to_csv(duration_file)

    return region_boundaries, data_filtered


def load_data(path: str) -> (np.ndarray, np.ndarray):
    data = np.loadtxt(path, delimiter=';', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def sort_data(x_pos: np.ndarray, y_pos: np.ndarray, eye_vectors: tuple) -> list:
    sorted_vectors = list()
    x_values = np.unique(x_pos)
    y_values = np.unique(y_pos)
    for x in x_values:
        indices_x = np.where(x_pos == x)[0]
        for y in y_values:
            indices_y = np.where(y_pos == y)[0]
            indices = np.intersect1d(indices_x, indices_y)
            vectors_x = list()
            vectors_y = list()
            for i in indices:
                vectors_x.append(eye_vectors[0][i])
                vectors_y.append(-eye_vectors[1][i])
            sorted_vectors.append((vectors_x, vectors_y))
    return sorted_vectors


def calculate_dist(data: list) -> list:
    dist = list()
    for i in data:
        d = list()
        x = np.array(i[0])
        y = np.array(i[1])
        for j in range(x.__len__()):
            d.append(np.sqrt(np.square(x[j]) + np.square(y[j])))
        dist.append(d)
    return dist


def remove_outliers(data: list) -> list:
    out = list()
    for i in data:
        d_x = np.array(i[0])
        d_y = np.array(i[1])

        # Determine bounds for outliers
        lb_x, ub_x = find_boundaries(d_x)
        lb_y, ub_y = find_boundaries(d_y)

        # remove outliers
        x = d_x[((d_x > lb_x) & (d_x < ub_x)) & ((d_y > lb_y) & (d_y < ub_y))]
        y = d_y[((d_x > lb_x) & (d_x < ub_x)) & ((d_y > lb_y) & (d_y < ub_y))]

        # remove non detected eyes
        x_n = x[(x != 0) & (y != 0)]
        y_n = y[(x != 0) & (y != 0)]
        out.append((x_n, y_n))
    return out


def remove_outliers_dist(data: list) -> list:
    out = list()
    for i in data:
        d_x = np.array(i)

        # Determine bounds for outliers
        lb_x, ub_x = find_boundaries(d_x)

        # remove outliers
        x = d_x[((d_x > lb_x) & (d_x < ub_x))]

        # remove non detected eyes
        x_n = x[x != 0]

        out.append(x_n)
    return out


def find_boundaries(l: np.ndarray) -> (int, int):
    q1 = np.quantile(l, .25)
    q3 = np.quantile(l, .75)
    irq = q3 - q1
    lb = q1 - 1.5 * irq
    ub = q3 + 1.5 * irq
    return lb, ub


# if __name__ == "__main__":
#     main()
