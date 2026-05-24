import multiprocessing
import queue

import matplotlib.pyplot as plt
import numpy as np

from dual_plot import process_lines


def process_functions(function_list, x_min=-10, x_max=10, debug=False):
    if debug:
        plot_queue = queue.Queue
        # show
        x = np.linspace(x_min, x_max, 400)
        dis_functions = []
        for func in function_list:
            if func[0] == 'mixed':
                c = func[5]  # Random inflection point
                a1, b1, a2, b2 = func[1], func[2], func[3], func[4]
                y = np.piecewise(x, [x < c, x >= c], [lambda x: a1 * x + b1, lambda x: a2 * x + b2])
                dis_functions.append((x, y))

            else:
                a, b = func[1], func[2]
                dis_functions.append((x, a * x + b))
        # Plot the first 10 functions as an example

        plt.figure(figsize=(10, 6))
        for i, (x, y) in enumerate(dis_functions):
            plt.plot(x, y, label=f'Function {i + 1}')

        plt.xlabel('x')
        plt.ylabel('f(x)')

    c_store = [func[5] for func in function_list if func[0] == 'mixed']
    c_store.sort()

    check_points = []

    for i in range(len(c_store) + 1):
        mini = c_store[i - 1] if i > 0 else x_min
        maxi = c_store[i] if i < len(c_store) else x_max
        funcs = []
        for func in function_list:
            func_type = func[0]
            if func_type == 'mixed':
                c = func[5]
                if c <= mini:
                    funcs.append((func[3], func[4]))
                else:
                    funcs.append((func[1], func[2]))
            else:
                funcs.append((func[1], func[2]))
        lines = np.array(funcs)
        check_points.append(process_lines(lines, mini, maxi, debug=debug))

    ret = min(check_points, key=lambda p: p[1])

    if debug:
        plt.plot(ret[0], ret[1], 'ro', markersize=10)

    return ret


def run(function_list, x_min, x_max, debug, pts_store):
    plt.figure(figsize=(10, 6))
    min_point = process_functions(function_list, x_min=x_min, x_max=x_max, debug=debug)
    print(f'Run prints ({x_min}, {x_max})')
    if pts_store is not None:
        for i in pts_store:
            plt.plot(x_min, i[0], 'bo', markersize=6)
            plt.plot(x_max, i[1], 'bo', markersize=6)
            plt.plot(i[2][0], i[2][1], 'bo', markersize=6)
    # plt.legend()
    plt.grid(True)
    plt.show()
    print(min_point)


def process_in_new_main(function_list, x_min=-10, x_max=10, debug=False, pts_store=None):
    process = multiprocessing.Process(target=run, args=(function_list, x_min, x_max, debug, pts_store))

    # Start the process
    process.start()


if __name__ == '__main__':

    # Seed for reproducibility
    np.random.seed(30)

    # Define the range for x
    x = np.linspace(0, 10, 400)


    def generate_functions(n):
        functions = []
        store = []
        for _ in range(n):
            # Randomly select the type of function (increasing, decreasing, or mixed)
            func_type = np.random.choice(['increasing', 'decreasing', 'mixed'])

            if func_type == 'increasing':
                a, b = np.random.rand() * 10, np.random.rand() * 10  # Random coefficients
                functions.append((x, a * x + b))
                store.append((func_type, a, b))

            elif func_type == 'decreasing':
                a, b = np.random.rand() * 10, np.random.rand() * 100 + 10  # Random coefficients
                functions.append((x, -a * x + b))
                store.append((func_type, -a, b))

            else:  # Mixed
                c = np.random.rand() * 10  # Random inflection point
                a1, b1, a2, b2 = np.random.rand(4) * 10  # Random coefficients
                b2 = (a1 + a2) * c + b1  # Adjust intercept to ensure the function doesn't go negative

                y = np.piecewise(x, [x < c, x >= c], [lambda x: a1 * x + b1, lambda x: -a2 * x + b2])
                functions.append((x, y))
                store.append((func_type, a1, b1, -a2, b2, c))

        return functions, store


    # Generate 100 random functions
    random_functions, store_functions = generate_functions(20)

    # Plot the first 10 functions as an example
    plt.figure(figsize=(10, 6))
    # for i, (x, y) in enumerate(random_functions[:20]):
    #     plt.plot(x, y, label=f'Function {i + 1}')
    #
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.title('First 10 Random Functions')

    print(process_functions(store_functions, debug=True))

    plt.legend()
    plt.grid(True)
    plt.show()
