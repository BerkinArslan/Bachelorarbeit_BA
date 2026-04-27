import numpy as np
import matplotlib.pyplot as plt

# years = np.linspace(2004, 2024, 21)
# passengers = np.array([75983, 74944, 78735, 79098, 82428, 81206, 82837, 89316, 93918,
#                        89450, 90978,91050,95465,95529,98161,100252,57787,57518,92313,101408,109135])
# plt.figure(figsize=(12, 8))
# plt.bar(years, passengers)
# plt.xlabel("Jahr", fontsize=14)
# plt.ylabel("Beförderte Passagiere (in Millionen)", fontsize=14)
# plt.xticks(years[::2], rotation=45, fontsize=12)
# plt.yticks(fontsize=12)
# plt.bar(years, passengers, color="steelblue")
# plt.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.show()

from rolland.database.rail.db_rail import UIC60
from utils import interpolate_contour_2d
rail_geometry = UIC60.rl_geo
rail_geometry = np.atleast_2d(rail_geometry)

# plt.figure()
# plt.plot(rail_geometry[:, 0], rail_geometry[:, 1], label='Originalkontur')
#
# # Ursprung markieren
# plt.scatter(0, 0, color='red', zorder=5)
# plt.text(0, 0, '  (0,0)', color='red')
#
# plt.xlabel('x-Koordinate [m]')
# plt.ylabel('y-Koordinate [m]')
# plt.title('Querschnittsgeometrie der UIC60-Schiene')
# plt.axis('equal')
# plt.grid(True)
# plt.legend(loc='upper left')
# plt.tight_layout
# plt.savefig('original_rail_plot', bbox_inches='tight')
# plt.show()
#
# # interpoliert
# rail_geometry = interpolate_contour_2d(rail_geometry, 25)
#
# plt.figure()
# plt.plot(rail_geometry[:, 0], rail_geometry[:, 1], label='Interpolierte Kontur')
#
# plt.scatter(0, 0, color='red', zorder=5)
# plt.text(0, 0, '  (0,0)', color='red')
#
# plt.xlabel('x-Koordinate [m]')
# plt.ylabel('y-Koordinate [m]')
# plt.title('Interpolierte Querschnittsgeometrie der UIC60-Schiene')
# plt.axis('equal')
# plt.grid(True)
# plt.tight_layout()
# plt.legend(loc='upper left')
# plt.savefig('interpolation_rail_plot_25', bbox_inches='tight')
#
# plt.show()
#
# rail_geometry = UIC60.rl_geo
# rail_geometry = np.atleast_2d(rail_geometry)
#
# plt.figure()
#
# # --- Original geometry (reference) ---
# plt.plot(
#     rail_geometry[:, 0],
#     rail_geometry[:, 1],
#     color='black',
#     linewidth=1.5,
#     alpha=0.6,
#     label='Originalkontur'
# )
#
# # --- Interpolations ---
# configs = [
#     (25, 'tab:blue', 'o'),
#     #(50, 'tab:orange', 's'),
#     #(100, 'tab:green', '^'),
# ]
#
# for n, color, marker in configs:
#     geom_interp = interpolate_contour_2d(rail_geometry, n)
#
#     plt.plot(
#         geom_interp[:, 0],
#         geom_interp[:, 1],
#         color=color,
#         linewidth=0.8,
#         alpha=0.8,
#     )
#
#     # markers to show discretization points
#     plt.scatter(
#         geom_interp[:, 0],
#         geom_interp[:, 1],
#         color=color,
#         s=10,
#         marker=marker,
#         alpha=0.8,
#         label=f'{n} Punkte-Interpolation'
#     )
#
# # Ursprung
# plt.scatter(0, 0, color='red', zorder=5)
# plt.text(0, 0, '  (0,0)', color='red')
#
# # Labels
# plt.xlabel('x-Koordinate [m]')
# plt.ylabel('y-Koordinate [m]')
# plt.title('Vergleich der Konturinterpolation')
#
# plt.axis('equal')
# plt.grid(True)
# plt.legend(loc='upper left')
#
# plt.tight_layout()
# plt.savefig('all_plots_original_and_25', bbox_inches='tight')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from utils import semi_circle_measurement_points

# --- rail geometry ---
rail_geometry = UIC60.rl_geo
rail_geometry = np.atleast_2d(rail_geometry)

# --- centre (same as simulation) ---
centre_3d = np.array([-0.71, -0.08, 75])  # z doesn't matter for plotting
radius = 2.5

def plot_setup(ax, title):
    ax.plot(rail_geometry[:, 0], rail_geometry[:, 1],
            color='black', linewidth=1.5, label='Schiene')

    # centre point
    ax.scatter(centre_3d[0], centre_3d[1], color='red', zorder=5)
    ax.text(
        centre_3d[0] - 0.1,
        centre_3d[1] + 0.1,
        'Mittelpunkt',
        color='red',
        ha='right',
        va='bottom'
    )

    # semicircle (visual only)
    theta = np.linspace(0, np.pi, 100)
    x = radius * np.cos(theta) + centre_3d[0]
    y = radius * np.sin(theta) + centre_3d[1]
    ax.plot(x, y, linestyle='--', color='gray', label='Halbkreis')

    ax.set_xlabel('x-Koordinate [m]')
    ax.set_ylabel('y-Koordinate [m]')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# =========================
# Frequency domain (multiple points)
# =========================
plot_setup(axes[0], 'Aufbau der Simulation im Frequenzbereich')

points_fd = semi_circle_measurement_points(
    centre_point=centre_3d,
    number_of_points=5,
    radius=radius
)

axes[0].scatter(points_fd[:, 0], points_fd[:, 1],
                color='blue', s=25, label='Beobachtungspunkte')

# =========================
# Time domain (single point)
# =========================
plot_setup(axes[1], 'Aufbau der Simulation im Zeitbereich')

points_td = semi_circle_measurement_points(
    centre_point=centre_3d,
    number_of_points=1,
    radius=radius
)

axes[1].scatter(points_td[:, 0], points_td[:, 1],
                color='green', s=40, label='Beobachtungspunkt')

# legends
axes[0].legend(loc='upper left')
axes[1].legend(loc='upper left')

plt.tight_layout()
plt.savefig('Simulationsaufbau', bbox_inches='tight')
plt.show()