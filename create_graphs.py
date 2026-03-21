import numpy as np
import matplotlib.pyplot as plt

years = np.linspace(2004, 2024, 21)
passengers = np.array([75983, 74944, 78735, 79098, 82428, 81206, 82837, 89316, 93918,
                       89450, 90978,91050,95465,95529,98161,100252,57787,57518,92313,101408,109135])
plt.figure(figsize=(12, 8))
plt.bar(years, passengers)
plt.xlabel("Jahr", fontsize=14)
plt.ylabel("Beförderte Passagiere (in Millionen)", fontsize=14)
plt.xticks(years[::2], rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.bar(years, passengers, color="steelblue")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()