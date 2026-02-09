import numpy as np
import matplotlib.pyplot as plt
from engine import differentiate

def calculate_p_in_y_time_domain(
        v: np.ndarray,
        rho: float,
        A: float,
        x: np.ndarray,
        y: np.ndarray,
):
    r = np.linalg.norm(y - x)
    t = np.linspace(0, 1, 16000 )
    dvdt = differentiate(v, t)
    p_yt_ = (rho/(4 * np.pi * r)) * A * dvdt
    return p_yt_

if __name__ == "__main__":
    frequency_spectrum = np.arange(1, 16001, 1)
    f0 = 1000
    sigma = 300
    v_fd = np.zeros_like(frequency_spectrum)
    v_fd = np.ones_like(frequency_spectrum)
    # v_fd[frequency_spectrum == 1000] = 1.0
    # v_fd[frequency_spectrum == 2000] = 0.9
    # v_fd[frequency_spectrum == 4000] = 0.8
    #v_fd = np.exp(-0.5 * ((frequency_spectrum - f0) / sigma) ** 2)
    # plt.plot(frequency_spectrum, v_fd)
    # plt.title("Frequency domain particle velocity")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Particle velocity (m/s)")
    #plt.show()
    v_td = np.fft.ifft(v_fd, 16000)
    p_yt = calculate_p_in_y_time_domain(v_td, 1.204, 1, np.array((0,0,0)), np.array((1,0,0)))
    p_yf = np.fft.fft(p_yt, 16000)
    plt.plot(frequency_spectrum, np.abs(p_yf))
    plt.title("Time domain monopole acoustic pressure in point y")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acoustic pressure (Pa)")
    plt.show()
    sound_in_db = np.log10(((np.abs(p_yf) + 2e-5)/ (2e-5))) * 20
    plt.plot(frequency_spectrum, sound_in_db)
    plt.title("Time domain monopole acoustic pressure in point y")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acoustic pressure (dB)")
    plt.show()