import numpy as np
from matplotlib import pyplot as plt

def calcualte_p_in_y_in_frequency_domain(v_fd: np.ndarray,
                                         frequency_spectrum:np.ndarray,
                                         x: np.ndarray,
                                         y:np.ndarray,
                                         A:float,
                                         c:float = 343.0,
                                         rho:float = 1.2041):
    k = 2 * np.pi * frequency_spectrum / c

    r = np.linalg.norm(y - x)

    p_yf = (np.exp(-1j*k*r)/(4 * np.pi * r)) * 1j  * 2 * np.pi * frequency_spectrum * rho * v_fd * A
    return p_yf

if __name__ == '__main__':
    A = 1,
    x = np.array((0,0,0))
    y = np.array((1,0,0))
    f0 = 1000
    sigma = 300

    frequency_spectrum = np.arange(1, 16001, 1)
    v_fd = np.zeros_like(frequency_spectrum)
    v_fd = np.ones_like(frequency_spectrum)
    # v_fd[frequency_spectrum == 1000] = 1.0
    # v_fd[frequency_spectrum == 2000] = 0.9
    # v_fd[frequency_spectrum == 4000] = 0.8
    # v_fd = np.exp(-0.5 * ((frequency_spectrum - f0) / sigma) ** 2)
    p_yf = calcualte_p_in_y_in_frequency_domain(v_fd, frequency_spectrum, x, y, A)
    plt.plot(frequency_spectrum, abs(p_yf))
    plt.title("Frequency Domain monopole acousic pressure in point y")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Pressure (Pa)")
    plt.show()
    sound_in_db = np.log10(((np.abs(p_yf) + 2e-5)/ (2e-5))) * 20
    plt.plot(frequency_spectrum, sound_in_db)
    plt.title("acoustic pressure in point y")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acoustic pressure (dB)")
    plt.show()