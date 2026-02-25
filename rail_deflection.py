from rolland import DiscrPad, Sleeper, Ballast, components, Discretization
from rolland.database.rail.db_rail import UIC60
from rolland import SimplePeriodicBallastedSingleRailTrack
from rolland import (
      PMLRailDampVertic,
      GaussianImpulse,
      DiscretizationEBBVerticConst,
      DeflectionEBBVertic)
from rolland.postprocessing import Response as resp, TDR
import numpy as np
import scipy as sp
from rolland import track, boundary


def rail_deflection_rolland(
        track: track.Track,
        boundary: PMLRailDampVertic,
        excitation,
)->tuple[np.ndarray, dict]:
    discretization = DiscretizationEBBVerticConst(
        track=track,
        bound=boundary,
    )

    deflection_results = DeflectionEBBVertic(
        discr=discretization,
        excit=excitation,
    )

    response = resp(results=deflection_results)

    deflection_information = {
        "nt": response.results.discr.nt,
        "dt": response.results.discr.dt,
        "T": response.results.discr.sim_t,
        'frequencies': sp.fft.fftfreq(response.results.discr.nt
                                      , response.results.discr.dt),
        'deflection': response.results.deflection,
        'nx': response.results.discr.nx,
        'dx': response.results.discr.dx,
    }

    return response, deflection_information

if __name__ == '__main__':
    # 1. TRACK DEFINITION ----------------------------------------------------------
    # Create a ballasted single rail track model with periodic supports
    track = SimplePeriodicBallastedSingleRailTrack(
        rail=UIC60,  # Standard UIC60 rail profile
        pad=DiscrPad(
            sp=[180e6, 0],  # Stiffness properties [N/m]
            dp=[18000, 0]  # Damping properties [Ns/m]
        ),
        sleeper=Sleeper(ms=150),  # Sleeper mass [kg]
        ballast=Ballast(
            sb=[105e6, 0],  # Ballast stiffness [N/m]
            db=[48000, 0]  # Ballast damping [Ns/m]
        ),
        num_mount=243,  # Number of discrete mounting positions
        distance=0.6  # Distance between sleepers [m]
    )

    # 2. SIMULATION SETUP ---------------------------------------------------------
    # Define boundary conditions (Perfectly Matched Layer absorbing boundary)
    boundary = PMLRailDampVertic(l_bound=33.0)  # 33.0 m boundary domain

    # Define excitation (Gaussian impulse between sleepers at 71.7m)
    excitation = GaussianImpulse(x_excit=71.7)

    resp_func, dict_func = rail_deflection_rolland(track, boundary, excitation)

    print(dict_func)

    # 3. DISCRETIZATION & SIMULATION ----------------------------------------------
    # Set up numerical discretization parameters
    discretization = DiscretizationEBBVerticConst(
        track=track,
        bound=boundary,
    )

    # Run the simulation and calculate deflection over time
    deflection_results = DeflectionEBBVertic(
        discr=discretization,
        excit=excitation
    )

    # 4. POSTPROCESSING & VISUALIZATION -------------------------------------------
    # 4.1 Calculate frequency response at excitation point
    response = resp(results=deflection_results)

    dt = response.results.discr.dt #time step information
    nt = response.results.discr.nt #number of time steps
    T = dt * nt
    #print(T)
    full_freqs = sp.fft.fftfreq(nt, dt) #numpy frequency steps
    #print(max(full_freqs))

    D = response.results.deflection #getting deflection information for rail and dampers
    nx = D.shape[0] // 2
    rail_defl = D[0::2, :nt] #masking it for only rail deflection
    #this data has the trac deflection of ith point at jth second as rail_defl[i, j]
    #print(rail_defl.shape)

    #...after this I should use this deflection for calcualtion of the monopoles