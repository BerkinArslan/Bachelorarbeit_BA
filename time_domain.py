import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
"""
Function naming convention:

<source_name>_<in_domain><in_type>__calc<calc_domain>__out<out_domain>
Source names: {monopole -> monopole, multi monopole -> monopole_matrix}
Domain names: {time domain -> t, frequency domain -> f}
Type names: {array -> a, callable -> f}
"""


def monopole_tf__calct_outf(
        v_td_fx: callable,
        time_full: np.ndarray,
        rho: float,
        A: float,
        x: np.array,
        y: np.array,
        delta_f: float,
)->np.array:
    """
    Calculates the sound pressure level in y.
    Input as time domain signal function
    Calculation in time domain
    Output as frequency domain signal
    :param v_td_fx: function of the time domain signal input
    :param rho: density of the air
    :param A: Assigned surface area of the monopole
    :param x: position of the monopole source
    :param y: position of the measurement point
    :param delta_f: frequency resolution
    :return: acoustic pressure in poin y in frequency domain
    """

    #Fast Fourier Settings:
    N = len(time_full)
    delta_t = time_full[1] - time_full[0]
    T = time_full[-1] + delta_t
    sampling_frequency = 1/delta_t
    frequency_spectrum = np.fft.fftfreq(N, d=delta_t)
    #distance
    r = np.linalg.norm(y - x)

    #calculating the derivative with scipy
    dvdt = sp.differentiate.derivative(lambda x: v_td_fx(x),
                                       time_full, initial_step=delta_t,)

    p_yt_ = (rho / (4 * np.pi * r)) * A * dvdt.df

    #transforming the result in frequency domain
    p_yf_ = sp.fft.fft(p_yt_, norm="forward")
    p_yf_ = p_yf_[:len(p_yf_)//2] * 2
    return p_yf_


def monopole_ta__calct__outf(
        v_fd:np.ndarray,
        rho:float,
        A:float,
        x:np.ndarray,
        y:np.ndarray,
        delta_f:float,
):
    """
    Calculates monopole source sound pressure level in y.
    Input as time domain signal array
    Calculation in time domain
    Output as frequency domain signal array
    :param v_td: particle velocity in time domain as array
    :param rho: air density
    :param A: assigned surface area of the monopole source
    :param x: position of the monopole source
    :param y: position of the acoustic pressure
    :param delta_f: frequency resolution
    :return: acoustic pressure in point y in frequency domain
    """

    #Fast Fourier Settings:
    N = len(v_fd)
    sampling_fequency = N * delta_f
    T = N/sampling_fequency
    time_full = np.linspace(0, T, int(N), endpoint=False)
    delta_t = T/N
    frequency_spectrum = sp.fft.fftfreq(N, d=delta_t)
    #distance
    r = np.linalg.norm(y - x)

    #calculating the derivative to t in frequency domain is easy
    #does not require numerical solutions
    dvdt = 1j * 2 * np.pi * frequency_spectrum * v_fd

    #changing back to the time domain to be able to calculate in time domain
    dvdt = sp.fft.ifft(dvdt, norm="forward")

    #calcualting
    p_yt_ = (rho / (4 * np.pi * r)) * A * dvdt

    #calculating frequency domain back
    p_yf_ = sp.fft.fft(p_yt_, norm="forward")

    #we only need the postive frequencies
    p_yf_ = p_yf_[:(len(frequency_spectrum)//2)]
    return p_yf_



