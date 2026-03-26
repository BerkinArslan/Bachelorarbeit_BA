import numpy as np
from rolland.database.rail.db_rail import UIC60
from matplotlib import pyplot as plt
import gmsh
import numpy as np
import scipy as sp


def interpolate_contour_2d(array, n)->np.ndarray:
    """
    Interpolate a 2d-contour array with n points.
    Works with non-equal contour point distances too.
    The created array has equal distances inbetween points.
    :param array: the 2d contour array.
    :param n: number of points in the end contour.
    :return: new 2d contour array.
    """
    array = np.atleast_2d(array)
    #np.diff calculates v_i+1 - v_i
    #np.linalg.norm calculates length of each v
    #IMPORTANT -> np.diff creates n-1 points
    len_intra_points = np.linalg.norm(np.diff(array, axis=0), axis=1)

    #len_intra_points create an array of length n-1
    #we need an array of length n
    contour_s = np.zeros(len(array))
    contour_s[1:] = np.cumsum(len_intra_points)

    total_length = contour_s[-1]

    new_contour_s = np.linspace(0, total_length, n)

    #now we interpolate x and y points separately
    x_new = np.interp(new_contour_s, contour_s, array[:,0])
    y_new = np.interp(new_contour_s, contour_s, array[:,1])

    new_array = np.stack([x_new, y_new], axis=1)
    return new_array


def create_mesh(
        rail_geometry: np.ndarray,
        mesh_size: float=None,
        L: float=None,
):
    """
    creates a mesh on the given contour and gives out the coordinates and indexes
    of the mesh triangles in form of array(N, 3, 3) and (N, 3)
    :param rail_geometry:
    :param mesh_size:
    :param L:
    :return:
    """

    #creating the flat base layer in 3d as z=0 for all points
    z_axis = np.zeros((rail_geometry.shape[0], 1))
    rail_geometry = np.concatenate((rail_geometry, z_axis), axis=1)
    gmsh.initialize()
    gmsh.model.add("rail")
    interpoint_distance = np.linalg.norm(rail_geometry[0] - rail_geometry[1])
    if mesh_size is None:
        mesh_size= interpoint_distance #this might be too small to calculate

    #creating the geometry
    point_tags = [
        gmsh.model.geo.addPoint(p[0],
                                p[1],
                                p[2], mesh_size)
        for p in rail_geometry
    ]

    line_tags = [
        gmsh.model.geo.addLine(point_tags[i],
                               point_tags[(i + 1) % len(point_tags)])
        for i in range(len(point_tags))
    ]

    #creating the surface
    curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.synchronize()

    #extrude
    if L is None:
        L = 1.0

    extruded = gmsh.model.geo.extrude(
        dimTags=[(2, surface)], dx=0, dy=0, dz=L
    )

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)

    #elementNodeTags include the indexes of triangles that are next to each other
    #in form of [n_0_1, n_0_2, n_0_3, n_1_1, n_1_2, n_1_3, ...]
    elementTypes, elementTags, elementNodeTags = gmsh.model.mesh.getElements(2)
    #to get it in triangle form
    triangle_tags = elementNodeTags[0].reshape(-1, 3)

    #we get element coordinates now
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    #to get it in [[x1y2z2],...] form...
    triangle_coords = nodeCoords.reshape(-1, 3)


    #elemntNodeTags have the order information. triangle_coordinates have the
    #coordinate infomration of each tag only once
    #nodeTags have the information of tag names from triangle_coords
    #So we need the indexes of the tags in nodeTags as in elementNodeTag
    #order so that we can call the index from triangle coords one by one
    tag_to_index = {tag: i for i, tag in enumerate(nodeTags)}
    triangle_index = np.vectorize(tag_to_index.get)(triangle_tags)

    return triangle_coords, triangle_index


def calculate_centre_and_area_triangles(
        triangle_coords: np.ndarray,
        triangle_index: np.ndarray,
):
    """
    Calculates the centre and the area of a triangle with the normal unit vector of the area
    :param triangle_coords: coordinate of each unique node (N, 1)
    :param triangle_index: index of the nodes that make up unique triangles (N, 3)
    :return:
                area: array of the area of the triangle
                centre: array of the centre of the triangle
                norm: array of the norm of the triangle
    """

    #calculate the area of each triangle:
    #linalg.norm(0,5 * (edge1 x edge2))
    #nodes
    p1 = triangle_coords[triangle_index[:, 0]]
    p2 = triangle_coords[triangle_index[:, 1]]
    p3 = triangle_coords[triangle_index[:, 2]]

    #edges
    edge_1 = p1 - p2
    edge_2 = p1 - p3

    #cross_product
    cp = np.cross(edge_1, edge_2)

    #area
    A = np.linalg.norm(cp, axis=1) * 0.5

    #now we calculate the centre
    centre = (p1 + p2 +p3) / 3.0

    #finally the normal vectors:
    norm = cp / np.linalg.norm(cp, axis=1, keepdims=True)

    return A, centre, norm


def calculate_projected_area(
        A: np.ndarray,
        norm: np.ndarray,
        axis: int | np.ndarray,
):
    """
    calculates the projected area of a triangle with the normal unit vector
    and the axis information for the projection
    :param Area: Area of the triangle
    :param norm: Normal vector of the triangle
    :param axis: axis of the projection
    :return:
                projected_area: projected area of the triangle
    """
    if isinstance(axis, int) and axis == 0:
        v = np.array((1, 0, 0))
    elif isinstance(axis, int) and axis == 1:
        v = np.array((0, 1, 0))
    elif isinstance(axis, int) and axis ==  2:
        v = np.array((0, 0, 1))
    else:
        #assume that v is not a unit vector:
        if np.linalg.norm(axis) == 0:
            raise ValueError("axis cannot be a zero vector")
        v = axis / np.linalg.norm(axis)

    A_projected = A * np.dot(norm, v)
    A_projected = np.abs(A_projected)

    return A_projected

def assign_v_to_points(
        rail_axis: np.ndarray,
        v_fd: np.ndarray, #v information for 1d
        centres: np.ndarray, #centres of the areas
        axis: int = 2
)-> np.ndarray:
    """
    v_fd gives one dimensional particle velocity information
    along the rail towards down. the centres in a crossection in rail
    will have the same particle velocity according to their axis along
    the rail. This function assigns the individual particle velocity.
    :param rail_axis: position information of the rail particle velocity    :param v_fd:  along the rail axis.
    :param centres: Coordinates of the centres of the rail-monopoles.
    :param axis: The axis of rail direction.
    :return: V_fd matrix for rail-monopole particle velocity.
    """
    z_axis = centres[:, axis]
    interpolated_v = sp.interpolate.interp1d(
        rail_axis,
        v_fd,
        axis=0,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate',
    )(z_axis)
    return interpolated_v

def semi_circle_measurement_points(
        centre_point: np.ndarray,
        number_of_points: int,
        radius: float,
)-> np.ndarray:

    theta = np.linspace(0, np.pi, number_of_points)
    x = radius * np.cos(theta) + centre_point[0]
    y = radius * np.sin(theta) + centre_point[1]
    z = np.full(number_of_points, centre_point[2])

    points = np.stack([x, y, z], axis=1)
    return points


def vector_deflection_total(
        u_x: np.ndarray,
        u_y: np.ndarray,
        phi_z: np.ndarray,
        centre: np.ndarray,
        rail_axis: int = 2,
        dx_of_displacement: np.float64 = 0.05,
)-> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the total lateral and vertical deflection for multiple
    monopole centres from given lateral, vertical, and rotating displacements.
    :param u_x: lateral displacement
    :param u_y: verticle displacement
    :param phi_z: rotating displacement
    :param centre: monopole centres
    :param rail_axis: the axis along which the rail direction is defined.
    :return: total lateral and vertical deflections of each monopole centre.
    """
    if not (u_y.shape == u_x.shape == phi_z.shape):
        raise ValueError("u_y, u_z and phi_x must have same shape")
    
    if u_y.ndim != 2:
        raise ValueError("Arrays must be 2D with shape (n_time, n_x)")
    
    n_t, n_x = u_y.shape
    N = centre.shape[0]

    #Interpolate along the rail axis so that the points
    #on the rail axis for each poisition is preserved

    monopole_z_positions = centre[:, rail_axis]
    given_displacement_z_positions = np.arange(0, n_x, 1) * dx_of_displacement
    u_x_interpolated = sp.interpolate.interp1d(
        given_displacement_z_positions,
        u_x,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate',
    )(monopole_z_positions)

    u_y_interpolated = sp.interpolate.interp1d(
        given_displacement_z_positions,
        u_y,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate',
    )(monopole_z_positions)

    phi_z_interpoalted = sp.interpolate.interp1d(
        given_displacement_z_positions,
        phi_z,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )(monopole_z_positions)
    #now we calculate the total displacement with the interpolated z

    u_x_total = u_x_interpolated + phi_z_interpoalted * centre[:, 1]
    u_y_total = u_y_interpolated + phi_z_interpoalted * centre[:, 0]

    return u_x_total, u_y_total

def run_simulation_semi_circle(
        circle_centre: np.ndarray,
        circle_radius: float,
        number_of_points: int,
        simulation_function: callable,
        multiply_by_two: bool = False,
        **simulation_kwargs,
):
    P_all = []
    measurement_points = semi_circle_measurement_points(
        circle_centre,
        number_of_points,
        circle_radius,
    )
    for point in measurement_points:
        P = simulation_function(**simulation_kwargs, y=point)
        P_all.append(np.abs(P) ** 2)

    P_mean = np.sqrt(np.mean(P_all, axis=0))
    return P_mean

#The next function is written by copilot!
def run_simulation_semi_circle_total(
        circle_centre: np.ndarray,
        circle_radius: float,
        number_of_points: int,
        simulation_function: callable,
        V_fd_z: np.ndarray,
        A_z: np.ndarray,
        V_fd_y: np.ndarray | None = None,
        A_y: np.ndarray | None = None,
        return_components: bool = False,
        **simulation_kwargs,
):
    """
    Same style as run_simulation_semi_circle, but supports vertical + lateral
    and returns RMS over semicircle points.

    If V_fd_y and A_y are given:
        total is computed coherently: p_total = p_z + p_y
    Else:
        behaves like vertical-only simulation.
    """

    P_total_all = []
    P_z_all = []
    P_y_all = []

    measurement_points = semi_circle_measurement_points(
        circle_centre,
        number_of_points,
        circle_radius,
    )

    for point in measurement_points:
        p_z = simulation_function(
            **simulation_kwargs,
            V_fd=V_fd_z,
            A=A_z,
            y=point,
        )
        P_z_all.append(np.abs(p_z) ** 2)

        if V_fd_y is not None and A_y is not None:
            p_y = simulation_function(
                **simulation_kwargs,
                V_fd=V_fd_y,
                A=A_y,
                y=point,
            )
            P_y_all.append(np.abs(p_y) ** 2)

            # Coherent total (phase-correct)
            p_total = p_z + p_y
            P_total_all.append(np.abs(p_total) ** 2)
        else:
            P_total_all.append(np.abs(p_z) ** 2)

    P_total_rms = np.sqrt(np.mean(P_total_all, axis=0))

    if return_components:
        P_z_rms = np.sqrt(np.mean(P_z_all, axis=0))
        if len(P_y_all) > 0:
            P_y_rms = np.sqrt(np.mean(P_y_all, axis=0))
        else:
            P_y_rms = None
        return P_total_rms, P_z_rms, P_y_rms

    return P_total_rms




if __name__ == "__main__":
    rail_geometry = UIC60.rl_geo
    rail_geometry = interpolate_contour_2d(rail_geometry, 100)
    triangle_coords, triangle_index = create_mesh(rail_geometry, 0.1, 1)
    print(triangle_index)
    print(triangle_coords)
    A, centre, norm = calculate_centre_and_area_triangles(triangle_coords, triangle_index)
    projected_area = calculate_projected_area(A, norm, 2)
    print(projected_area)

    circle = semi_circle_measurement_points(np.array((0,0,0)), 10, 1)
    for point in circle:
        print(point)




