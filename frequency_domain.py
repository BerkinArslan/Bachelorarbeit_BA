import numpy as np
from matplotlib import pyplot as plt

def calcualte_p_in_y_in_frequency_domain(v_fd: np.ndarray,
                                         frequency_spectrum:np.ndarray,
                                         x: np.ndarray,
                                         y:np.ndarray,
                                         A:float,
                                         c:float = 343.0,
                                         rho:float = 1.2041):
    """
    Calculates the single monopole sound pressure from the source in point y.
    :param v_fd: Array of monopole partcile speed in the source in frequency domain.
    :param frequency_spectrum: The spectrum of the frequency domain.
    :param x: The monopole source point in x,y,z axis.
    :param y: The Measurement point in x,y,z axis.
    :param A: Assigned monopole surface area.
    :param c: The speed of sound of the medium (Standard 343m/s).
    :param rho: The density of the medium (Standard 1.2041 kg/m^3).
    :return: p(f) Frequency dependent sound pressure array.
    """
    k = 2 * np.pi * frequency_spectrum / c

    r = np.linalg.norm(y - x)

    p_yf = (np.exp(-1j*k*r)/(4 * np.pi * r)) * 1j  * 2 * np.pi * frequency_spectrum * rho * v_fd * A
    return p_yf

def monopole_multi_fa__calcf__outf(
        V_fd: np.ndarray,
        freqs: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        A: float,
        c: float = 343.0,
        rho: float = 1.2041,
)->np.ndarray:
    """
    Calculates the sound pressure in measurement point y from multiple
    monopole sources in matrix X.
    :param V_fd: Frequency dependent particle speed matrix of the points in X.
    :param freqs: spectrum of the frequency domain.
    :param X: Coordinates of the source points (cartesian).
    :param y: Coordinate of the measurement point (cartesian).
    :param A: Assigned monopole surface area matrix of the sources.
    :param c: The speed of sound of the medium (Standard 343m/s).
    :param rho: The density of the medium (Standard 1.2041 kg/m^3).
    :return: The frequency dependent sound pressure array in measurement point y.
    """

    k = 2 * np.pi * freqs / c
    k = k[None, :]

    r = np.linalg.norm(y - X, axis=1)
    r = r[:, None]
    if np.any(r == 0):
        raise ValueError("Measurement point cannot be the same as a monopole source!")

    omega = 2 * np.pi * freqs
    omega = omega[None, :]
    p = np.exp(-1j * k * r)/(4 * np.pi * r) * 1j * omega * V_fd * rho * A
    p = np.sum(p, axis=0)
    return p

if __name__ == '__main__':
    A = 1,
    x = np.array((0,0,0))
    y = np.array((1,0,0))


    frequency_spectrum = np.arange(1, 8001, 1)
    v_fd = np.zeros_like(frequency_spectrum)
    v_fd[frequency_spectrum == 1000] = 1.0
    # p_yf = calcualte_p_in_y_in_frequency_domain(v_fd, frequency_spectrum, x, y, A)
    # plt.plot(frequency_spectrum, abs(p_yf))
    # plt.title("Frequency Domain monopole acousic pressure in point y")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Pressure (Pa)")
    # plt.show()
    # sound_in_db = np.log10(((np.abs(p_yf) + 2e-5)/ (2e-5))) * 20
    # plt.plot(frequency_spectrum, sound_in_db)
    # plt.title("acoustic pressure in point y")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Acoustic pressure (dB)")
    # plt.show()

    X = np.array((x,x))
    V_fd = np.array((v_fd,v_fd))
    print(monopole_multi_fa__calcf__outf(V_fd, frequency_spectrum, X, y, A))