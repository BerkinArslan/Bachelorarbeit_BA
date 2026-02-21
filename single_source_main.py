from frequency_domain import calcualte_p_in_y_in_frequency_domain
from time_domain import calculate_p_in_time_domain, calculate_p_in_time_domain_from_frequency_domain_signal
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
#TODO: Small mistake discovered:
# the fft functions assume index 0 -> 0 Hz.
# the time domain function assumes index 0 -> 1Hz

def sin_function(f, t, A=1):
    v_td_fx = A * np.sin(2 * np.pi * f * t)
    return v_td_fx

def gaussian_distribution(f, s, all_f, A=1, add_random_phase=True):
    v_fd_fx = A * (
            np.exp(-(all_f - f) ** 2 / (2 * s ** 2)) +
            np.exp(-(all_f + f) ** 2 / (2 * s ** 2))
    )

    if add_random_phase:
        phase = np.random.uniform(-np.pi, np.pi, len(v_fd_fx))
        # enforce symmetry
        phase = (phase - phase[::-1]) / 2
        v_fd_fx = v_fd_fx * np.exp(1j * phase)

    return v_fd_fx

def gaussian_distribution_time_domain(f, s, all_f, A=1):
    v_fd_fx = gaussian_distribution(f, s, all_f, A)
    v_td_fx = sp.fft.ifft(v_fd_fx)
    return v_td_fx

if __name__ == '__main__':

    #Global source variables:
    A = 1 #m^2
    rho = 1.2041 #kg/m^3
    c = 343.0 #m/s
    spectrum_min = 1 #Hz
    spectrum_max = 8000 #Hz

    #frequency domain source
    #particle velocity array:
    v_fd = np.zeros_like(np.arange(spectrum_min, spectrum_max+1, 1))
    v_fd[1000] = 1.0
    p_y_fd = calcualte_p_in_y_in_frequency_domain(
        v_fd=v_fd,
        frequency_spectrum=np.arange(spectrum_min, spectrum_max+1, 1),
        x=np.array((0,0,0)),
        y=np.array((1,0,0)),
        A=A,
        c=c,
        rho=rho
    )
    #2e-5 to zero the negative dB. they are not relevant for the human ear.
    p_y_fd_db = np.log10(((np.abs(p_y_fd) + 2e-5) / 2e-5)) * 20


    #time domain source
    #particle velocity function
    v_td_fx = lambda x: sin_function(1000, x, A=A)

    p_y_td = calculate_p_in_time_domain(
        v_td_fx=v_td_fx,
        rho=rho,
        A=A,
        x=np.array((0,0,0)),
        y=np.array((1,0,0)),
        delta_f=1,
        spektrum=(spectrum_min, spectrum_max)
    )

    p_y_td_db = np.log10((abs(p_y_td) + 2e-5)/2e-5) * 20

    #plotting pressure in pa:
    frequency_spektrum = np.arange(spectrum_min, spectrum_max+1, 1)
    plt.plot(frequency_spektrum, np.abs(p_y_fd),
             label="Calculation in frequency domain",
             color="blue",
             alpha=0.5,
             linestyle=":",
             marker="o",
             markevery=1000,)
    plt.plot(frequency_spektrum, np.abs(p_y_td), label="Calculation in time domain",
             color="red",
             alpha=0.5,
             linestyle="--",
             marker="x",
             markevery=1000,)
    plt.title("Single Monopole Source Calculation Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acoustic pressure (pa)")
    plt.legend()
    plt.show()

    #plotting pressure in dB
    plt.plot(frequency_spektrum, np.abs(p_y_td_db),
             label="Calculation in time domain",
             color="blue",
             alpha=0.5,
             linestyle=":",
             marker="o",
             markevery=1000,)
    plt.plot(frequency_spektrum, np.abs(p_y_fd_db),
             label="Calculation in frequency domain",
             color="red",
             alpha=0.5,
             linestyle="--",
             marker="x",
             markevery=1000,)
    plt.title("Single Monopole Source Calculation Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acoustic pressure (dB)")
    plt.legend()
    plt.show()

    f_axis = sp.fft.fftfreq(16000, d=1 / 16000)
    v_fd_fx = gaussian_distribution(1500, 100, f_axis, A=0.707)
    p_fd_from_td = calculate_p_in_time_domain_from_frequency_domain_signal(
        v_fd_fx=v_fd_fx,
        rho=rho,
        A=A,
        x=np.array((0,0,0)),
        y=np.array((1,0,0)),
        delta_f=1,
        spektrum=(spectrum_min, spectrum_max)
    )
    p_fd_from_fd = calcualte_p_in_y_in_frequency_domain(v_fd_fx[:8000],
                                                        frequency_spectrum=np.arange(spectrum_min, spectrum_max + 1, 1),
                                                        x=np.array((0, 0, 0)),
                                                        y=np.array((1, 0, 0)),
                                                        A=A,
                                                        c=c,
                                                        rho=rho
                                                        )

    p_fd_from_td = np.log10((abs(p_fd_from_td) + 2e-5)/2e-5) * 20
    p_fd_from_fd = np.log10((abs(p_fd_from_fd) + 2e-5) / 2e-5) * 20
    plt.plot(frequency_spektrum, np.abs(p_fd_from_td),
             label="Calculation in time domain",
             color="blue",
             alpha=0.5,
             linestyle=":",
             marker="o",
             markevery=1000,
             )
    plt.plot(frequency_spektrum, np.abs(p_fd_from_fd),
             label="Calculation in frequency domain",
             color="red",
             alpha=0.5,
             linestyle="--",
             marker="x",
             markevery=1000,)
    plt.title("Single Monopole Source Calculation Comparison for Gaussian Distribution")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acoustic pressure (dB)")
    plt.legend()
    plt.show()
    a = np.sum(np.abs(p_fd_from_td) ** 2)
    b = np.sum(np.abs(p_fd_from_fd) ** 2)
    print(a/b)


