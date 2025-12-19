import numpy as np
import glob

def orderparameter(orientation):
    vec = np.array([np.cos(orientation), np.sin(orientation)])
    cos = vec[0, :, :]
    sin = vec[1, :, :]

    cosS = np.sum(vec[0, :, :], axis=1)
    sinS = np.sum(vec[1, :, :], axis=1)

    mean = np.arctan2(sinS, cosS)

    cosdtheta = cos * np.cos(mean)[:, None] + sin * np.sin(mean)[:, None]
    cos2_Em = np.mean(cosdtheta**2, axis=1)
    S = 2*(cos2_Em - 1/2)

    return S

def emsembleS(folder):
    files = glob.glob(f"{folder}/seed*")

    S_lists = []

    for file in files:
        orientation = np.load(f"{file}/orientations_history.npy")
        S = orderparameter(orientation)
        S_lists.append(S)
    S_array = np.array(S_lists)

    return S_array

#if __name__ == "__main__":