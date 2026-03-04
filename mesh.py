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
elementTypes, elementTags, elementNodeTags = gmsh.model.mesh.getElements(dim=2)

nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
coords = nodeCoords.reshape(-1,3)

tag_to_index = {tag: i for i, tag in enumerate(nodeTags)}

monopole_area = np.zeros(len(nodeTags))

nodes = elementNodeTags[0].reshape(-1,3)

for elem_nodes in elementNodeTags[0].reshape(-1,3):

    i0 = tag_to_index[elem_nodes[0]]
    i1 = tag_to_index[elem_nodes[1]]
    i2 = tag_to_index[elem_nodes[2]]

    p0 = coords[i0]
    p1 = coords[i1]
    p2 = coords[i2]

    v1 = p1 - p0
    v2 = p2 - p0

    normal = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(normal)

    n_unit = normal / (2*area)

    # projected area downward (onto xy plane)
    projected = area * abs(n_unit[2])

    # distribute equally to triangle nodes
    monopole_area[i0] += projected / 3
    monopole_area[i1] += projected / 3
    monopole_area[i2] += projected / 3


gmsh.finalize()
