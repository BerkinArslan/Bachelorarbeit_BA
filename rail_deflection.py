from rolland import DiscrPad, Sleeper, Ballast
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
full_freqs = sp.fft.fftfreq(nt, dt) #numpy frequency steps
print(max(full_freqs))

D = response.results.deflection #getting deflection information for rail and dampers
nx = D.shape[0] // 2
rail_defl = D[0::2, :nt] #masking it for only rail deflection
#this data has the trac deflection of ith point at jth second as rail_defl[i, j]
print(rail_defl)

#...after this I should use this deflection for calcualtion of the monopoles