from rolland.database.rail.db_rail import UIC60
from matplotlib import pyplot as plt
import gmsh
import numpy as np
from utils import interpolate_contour_2d

rail_geometry = UIC60.rl_geo
rail_geometry = interpolate_contour_2d(rail_geometry, 100)
x_points = [point[0] for point in rail_geometry]
y_points = [point[1] for point in rail_geometry]
plt.plot(x_points, y_points)
plt.show()

distance_points = np.linalg.norm(np.array(rail_geometry[1])
                                 - np.array(rail_geometry[2]))
print(distance_points)

gmsh.initialize()
gmsh.model.add("rail")

point_tags = []
for p in rail_geometry:
    tag = gmsh.model.geo.addPoint(p[0], p[1], 0, distance_points)
    point_tags.append(tag)

line_tags = []
for i in range(len(point_tags)):
    start_tag = point_tags[i]
    end_tag = point_tags[(i+1)%len(point_tags)]
    line_tags.append(gmsh.model.geo.addLine(start_tag, end_tag))

cl = gmsh.model.geo.addCurveLoop(line_tags)
surface = gmsh.model.geo.addPlaneSurface([cl])

L = 10.0
gmsh.model.geo.extrude([(2, surface)], 0, 0, L)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

gmsh.write("rail.msh")
gmsh.finalize()
