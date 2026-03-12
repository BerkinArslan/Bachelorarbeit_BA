from rolland import DiscrPad, Sleeper, Ballast
from rolland.database.rail.db_rail import UIC60
from rolland import SimplePeriodicBallastedSingleRailTrack
from rolland import (
      PMLRailDampVertic,
      GaussianImpulse,
      DiscretizationEBBVerticConst,
      DeflectionEBBVertic)
from rolland.postprocessing import Response as resp
import numpy as np
import scipy as sp
from rolland import track

from frequency_domain import monopole_multi_fa__calcf__outf
from rail_deflection import rail_deflection_rolland
from time_domain import monopole_multi_ta__calct__outf
from utils import interpolate_contour_2d, create_mesh, calculate_centre_and_area_triangles, calculate_projected_area, \
    assign_v_to_points, semi_circle_measurement_points
from matplotlib import pyplot as plt

# 1. TRACK DEFINITION ----------------------------------------------------------
# Create a ballasted single rail track model with periodic supports
track = SimplePeriodicBallastedSingleRailTrack(
    rail=UIC60,  # Standard UIC60 rail profile
    pad=DiscrPad(
        sp=[4.5e8, 0],  # Stiffness properties [N/m]
        dp=[33000, 0]  # Damping properties [Ns/m]
    ),
    sleeper=Sleeper(ms=150),  # Sleeper mass [kg]
    ballast=Ballast(
        sb=[105e6, 0],  # Ballast stiffness [N/m]
        db=[75000, 0]  # Ballast damping [Ns/m]
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

#now we create the mesh and calcualte the projected area of each monopole and its position

rail_geometry = UIC60.rl_geo
rail_geometry = interpolate_contour_2d(rail_geometry, 100)
triangle_coords, triangle_index = create_mesh(rail_geometry,
                                              mesh_size=10.0,
                                              L=146.0 #146.0
                                              )

A, centre, norm = calculate_centre_and_area_triangles(triangle_coords, triangle_index)
print(f'mean area = {A.mean()}')
projected_area = calculate_projected_area(A, norm, 1)
print(f'projected_area mean = {projected_area.mean()}')
nt = dict_func['nt']
dt = dict_func['dt']
deflection = dict_func['deflection'][0::2, :nt]
full_freq_spectrum = sp.fft.fftfreq(nt, dt)

f_axis_sim = dict_func['frequencies']
mask = (full_freq_spectrum > f_axis_sim[0]) & (full_freq_spectrum < f_axis_sim[-1])

deflection_fd = sp.fft.fft(deflection, norm='forward', axis=1) * 2

omega = 2 * np.pi * f_axis_sim

rail_v_fd = (1j * omega[None, :]) * deflection_fd
rail_v_td = np.gradient(deflection, dt, axis=1)


nx = dict_func['nx']
dx = dict_func['dx']
x_axis = np.arange(nx) * dx
L = nx * dx
print(L)

#now we have the centres, the projected areas and the v of each z position.
#we have to now create an array of the v for each centre position
#maybe I should write a function in the utils modul to assign the v
#for each function.

triangle_v_fd = assign_v_to_points(
    x_axis,
    rail_v_fd,
    centre
)

triangle_v_td = assign_v_to_points(
    x_axis,
    rail_v_td,
    centre
)

print(triangle_v_fd.shape)

measurement_points = semi_circle_measurement_points(
    np.array((-2, 0, 75)),
    1,
    2.5
)
P_all = []
# for point in measurement_points:
#
#     P = monopole_multi_fa__calcf__outf(triangle_v_fd,
#                                        f_axis_sim,
#                                        centre,
#                                        point,
#                                        projected_area[:, None],) * 2
#     P_all.append(np.abs(P) ** 2 )

P_all = []
P_all_td = []
z_axis_points = np.linspace(73, 73, 1)
for point_z in z_axis_points:
    measurement_points = semi_circle_measurement_points(
        np.array((-2, 0, point_z)),
        5,
        2.5
    )
    for point in measurement_points:
        P = monopole_multi_fa__calcf__outf(triangle_v_fd,
                                           f_axis_sim,
                                           centre,
                                           point,
                                           projected_area[:, None],)
        p_fd_from_ta = monopole_multi_ta__calct__outf(
                                            triangle_v_td,
                                            centre,
                                            point,
                                            dt,
                                            projected_area[:, None],
                                        ) * 2 #multiplied by 2 because inside fft with no correction
        P_all.append(np.abs(P) ** 2 )
        P_all_td.append(np.abs(p_fd_from_ta) ** 2)


P_mean = np.mean(P_all, axis=0)
P_mean_td = np.mean(P_all_td, axis=0)
with open("Default Dataset.csv", "r") as f:
    content = f.read().replace(",", ".")
with open("Default Dataset.csv", "w") as f:
    f.write(content)
wagner_simulation_data = np.genfromtxt('Default Dataset.csv', delimiter=';')
print(wagner_simulation_data.shape)
print(wagner_simulation_data[:, 0])
print(wagner_simulation_data[:, 1])

p0 = 2e-5
P_db = 20 * np.log10((np.sqrt(P_mean) + p0)/ p0)
P_td_db = 20 * np.log10((np.sqrt(P_mean_td) +p0)/ p0)
plt_mask = (f_axis_sim >= 100) & (f_axis_sim <= 1500)
plt_f_axis = f_axis_sim[plt_mask]
plt_P_db = P_db[plt_mask]
plt_P_td_db = P_td_db[plt_mask]
#plt.plot(f_axis_sim[:len(f_axis_sim)//2], P_db)
plt.plot(plt_f_axis, plt_P_db, label="frequency domain calculation")
plt.plot(plt_f_axis, plt_P_td_db, label="time domain calculation")
plt.plot(wagner_simulation_data[:, 0], wagner_simulation_data[:, 1], label="wagner simulation")
plt.legend()
plt.xscale('log')
plt.show()