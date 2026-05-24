import copy

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from shapely.geometry import LineString


def process_lines(input_lines, x_min=-15, x_max=15, debug=False):
    points = np.copy(input_lines)
    points[:, 1] *= -1

    # Sort points based on x-coordinate
    points = points[np.argsort(points[:, 0])]

    # Compute the convex hull
    hull = ConvexHull(points)

    def check_elements(arr1, arr2):
        return np.all(np.isin(arr1, arr2))

    def point_to_dual_line(point):
        m, b = point[0], -point[1]
        # draw_line(m, b)
        return m, b

    def find_index(lst, num):
        for idx, item in enumerate(lst):
            if item == num:
                return idx
        return -1

    def draw_line(m, b):
        x = np.linspace(x_min, x_max, 400)
        y = m * x + b
        plt.plot(x, y, 'k-')

    def modify_list(lst, num):
        try:
            idx = find_index(lst, num)  # Step 1: Find the index of the number
            left_slice = lst[:idx]  # Step 2: Slice the left part of the index
            right_slice = lst[idx:]
            lst = np.append(right_slice, left_slice)  # Step 3: Append the left slice to the right
        except ValueError:
            print(f"{num} not found in the list.")
        return lst

    hull_vertices = copy.deepcopy(hull.vertices)

    hull_vertices = modify_list(hull_vertices, 0)

    last_index = find_index(hull_vertices, len(points) - 1)  # Step 1: Find the index of the number

    left_slice = hull_vertices[:last_index]
    right_slice = hull_vertices[last_index:]

    right_slice = np.append(right_slice, [0])
    left_slice = np.append(left_slice, [len(points) - 1])

    # Plotting
    # plt.plot(points[:, 0], points[:, 1], 'o')  # Plot points

    def find_intersection(m1, c1, m2, c2):
        # If the slopes are the same, the lines are parallel
        if m1 == m2:
            return None  # No intersection, unless they're the same line

        xi = (c2 - c1) / (m1 - m2)
        yi = m1 * xi + c1

        return (xi, yi)

    def line_equation(P1, P2):
        x1, y1 = P1
        x2, y2 = P2

        # Calculate slope
        m = (y2 - y1) / (x2 - x1)

        # Calculate y-intercept
        c = y1 - m * x1

        return m, c

    upper = []

    for simplex in hull.simplices:
        if check_elements(simplex, left_slice):
            # plt.plot(points[simplex, 0], points[simplex, 1], 'r-')  # Plot hull edges
            upper.append(points[simplex])
        # else:
        #     plt.plot(points[simplex, 0], points[simplex, 1], 'b-')  # Plot hull edges

    upper = [((line[0][0], line[0][1]), (line[1][0], line[1][1])) for line in upper]

    end_points = []

    # for lines in input_lines:
    #     draw_line(lines[0], lines[1])

    check_points = []
    starting_x = -1000
    for id in range(len(left_slice)):
        point_id = left_slice[id]
        point = points[point_id]
        end_pt = 1000
        if id < len(left_slice) - 1:
            next_point = points[left_slice[id + 1]]
            x0, y0 = find_intersection(point[0], -point[1], next_point[0], -next_point[1])
            end_pt = x0
        if starting_x < x_min:
            starting_x = x_min
        if end_pt > x_max:
            end_pt = x_max
        if starting_x <= end_pt:
            check_points.append((starting_x, point[0] * starting_x - point[1]))
            check_points.append((end_pt, point[0] * end_pt - point[1]))
            x = np.linspace(starting_x, end_pt, 400)
            y = point[0] * x - point[1]
            if debug:
                plt.plot(x, y, 'r-', linewidth=3)
        if id < len(left_slice) - 1:
            starting_x = x0

    # for lines in upper:
    #     m, c = line_equation(lines[0], lines[1])
    #     plt.scatter(m, -c, color='red', zorder=5)
    #     end_points.append((m, -c))

    # for simplex in hull.simplices:
    #     plt.plot(points[simplex, 0], points[simplex, 1], 'r-')  # Plot hull edges
    ret = min(check_points, key=lambda p: p[1])
    return ret


if __name__ == '__main__':
    np.random.seed(40)  # For reproducibility
    # Generate 20 random points
    gen_points = np.random.rand(40, 2)

    gen_lines = np.copy(gen_points)
    gen_lines[:, 1] *= -1
    process_lines(gen_lines, x_min=-15, x_max=0)
    plt.axis('equal')
    plt.show()
