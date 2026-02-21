import numpy as np
import matplotlib.pyplot as plt
import scipy as sp



def calculate_p_in_time_domain(
        v_td_fx: callable,
        rho: float,
        A: float,
        x: np.ndarray,
        y: np.ndarray,
        delta_f: float,
        spektrum: tuple,
        )->np.ndarray:
    """
    Calculates the frequency domain sound pressure level in y
    from the time domain monopole source with v
    :param v_td_fx: function of particle speed v in time domain
    :param rho: air density
    :param A: assigned surface area of the monopole source
    :param x: position of the monopole source
    :param y: recoding position of the acoustic pressure
    :param delta_f: resolution of the resulting acoustic pressure
    :param spektrum: frequency sprektrum that is targetted to be
    calculated for the acosutic pressure in y
    :return: acoustic pressure in point y in frequency domain
    """

    #number of discrete time points needed for the resolution
    #and the frequency spectrum wanted
    N = np.ceil(2 * spektrum[1]/delta_f)
    #total time of the measurement needed
    T = N/( 2 * spektrum[1])
    #time points of the measurement
    time_full = np.linspace(0, T, int(N), endpoint=False)
    frequency_spectrum_real = np.linspace(spektrum[0], spektrum[1], int(N/2))
    #particle velocity in measurement points
    v_td = v_td_fx(time_full)
    #distance of y to x (microphone to monopole source)
    r = np.linalg.norm(y - x)
    #2 different differentiation methods.
    #scipy is delivering better results
    #one is scipy using the particle velocity function
    dvdt = sp.differentiate.derivative(lambda x: v_td_fx(x),
     time_full, initial_step=time_full[1]-time_full[0])
    p_yt_ = (rho / (4 * np.pi * r)) * A * dvdt.df
    #one is numpy, uses the particle velocity points
    '''dvdt = np.gradient(v_td, time_full[1] - time_full[0], edge_order=2)
    p_yt_ = (rho / (4 * np.pi * r)) * A * dvdt'''

    from scipy.signal.windows import blackman
    #windowing function is not necessary with scipy derivative
    #and the result is better with it.
    #w = blackman(int(N))

    #discrete fourier transform.
    p_yf_ = sp.fft.fft(p_yt_, norm="forward")

    #to account for negative frequencies halving the amplitudes,
    #pressure is doubled.
    p_yf_ = p_yf_[:len(frequency_spectrum_real)] * 2
    # plt.plot(frequency_spectrum_real, abs(p_yf_[:len(frequency_spectrum_real)]))
    # plt.show()
    return p_yf_

def calculate_p_in_time_domain_from_frequency_domain_signal(
        v_fd_fx: np.ndarray,
        rho: float,
        A: float,
        x: np.ndarray,
        y: np.ndarray,
        delta_f: float,
        spektrum: tuple,
        )->np.ndarray:
    """

    :param v_fd_fx:
    :param rho:
    :param A:
    :param x:
    :param y:
    :param delta_f:
    :param spektrum:
    :return:
    """
    # number of discrete time points needed for the resolution
    # and the frequency spectrum wanted
    N = np.ceil(2 * spektrum[1] / delta_f)
    # total time of the measurement needed
    T = N / (2 * spektrum[1])
    # time points of the measurement
    time_full = np.linspace(0, T, int(N), endpoint=False)
    dt = time_full[1] - time_full[0]
    frequency_spectrum_all = sp.fft.fftfreq(len(v_fd_fx), d=dt)
    frequency_spectrum_real = frequency_spectrum_all[:int(N)//2]
    dv_fd_fxdt = 1j * 2 * np.pi * frequency_spectrum_all * v_fd_fx
    dvdt = sp.fft.ifft(dv_fd_fxdt, norm="forward")
    r = np.linalg.norm(y - x)
    p_yt_ = (rho / (4 * np.pi * r)) * A * dvdt
    p_yf_ = sp.fft.fft(p_yt_, norm="forward")
    p_yf_ = p_yf_[:len(frequency_spectrum_real)] * 2
    return p_yf_


if __name__ == "__main__":

    def sin_function(f, t, A=1):
        v_td_fx = A * np.sin(2 * np.pi * f * t)
        return v_td_fx

    p = calculate_p_in_time_domain(
        v_td_fx=lambda x: sin_function(1000, x),
        rho=1.2041,
        A=1.0,
        x = np.array((0,0,0)),
        y= np.array((1,0,0)),
        delta_f=1.0,
        spektrum=(1,8000),
    )

    p_db = np.log10((abs(p[:8000]) + 2e-5)/2e-5) *20
    frequencies = np.linspace(1, 8000, 8000)
    plt.plot(frequencies, p_db, label="p_(y) in dB from x with v_x = sin(1000Hz*2*pi)m/s")
    plt.title("acoustic pressure in point y, calculated in time domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Acoustic pressure (dB)")
    plt.legend(loc = "upper right")
    plt.show()



