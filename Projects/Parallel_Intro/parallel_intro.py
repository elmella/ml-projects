# iPyParallel - Intro to Parallel Programming
from ipyparallel import Client
import numpy as np
import time
import matplotlib.pyplot as plt
# Problem 1
def prob1():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    # Create the client and direct view
    client = Client()
    dview = client[:]
    # Import scipy.sparse as sparse on all engines
    dview.execute("import scipy.sparse as sparse")
    client.close()
    # return the direct view
    return dview
# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    # Create the client and direct view
    client = Client()
    dview = client[:]
    dview.block = True
    dview.push(dx)
    # Pull the variables back and make sure they haven't changed
    new_dx = {}
    for key in dx.keys():
        new_dx[key] = dview.pull(key)[0]
    client.close()
    # return True if the dictionaries are the same
    return dx == new_dx

# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    # define the function
    f = lambda n: np.random.normal(size = n)

    # Create the client and direct view
    client = Client()
    dview = client[:]
    dview.block = True
    
    # Push the function to the engines
    dview.execute("import numpy as np")
    responses = dview.apply_async(f, n)

    # Gather the results
    results = [r for r in responses]
    client.close()

    # return the results
    return [i.mean() for i in results], [i.min() for i in results], [i.max() for i in results]
    

# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    # declare constants
    n_array = [1000000, 5000000, 10000000, 15000000]
    N = 8 # numper of CPU cores
    parallel_time = []
    serial_time = []
    for n in n_array:
        # time parallel
        start = time.time()
        prob3(n)
        end = time.time()
        parallel_time.append(end - start)

        # time serial
        start = time.time()
        for _ in range(N):
            prob3(n)
        end = time.time()
        serial_time.append(end - start)

    # Plot the results
    plt.plot(n_array, parallel_time, label = "Parallel")
    plt.plot(n_array, serial_time, label = "Serial")

    plt.title("Parallel and Serial Execution Time")
    plt.ylabel("t (sec)")
    plt.xlabel("n")
    plt.legend()
    plt.show()


# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    # get the x_vals, h, and define the trap function
    x_vals = np.linspace(a, b, n)
    h = (b - a)/(n - 1)
    trap = lambda x: (h/2)*(f(x[0]) + f(x[-1])) + h*np.sum(f(x[1:-1]))
    
    # number of cores
    N = 8

    # Create the client and direct view
    client = Client()
    dview = client[:]
    dview.block = True
    # Push the function and the trapezoidal rule to the engines
    dview.execute("import numpy as np")
    dview.push({"f": f, "h": h, "trap": trap})
    
    # partition the x_vals into N parts
    x_1 = x_vals[:n//N + 1]
    x_2 = x_vals[n//N:2*n//N + 1]
    x_3 = x_vals[2*n//N:3*n//N + 1]
    x_4 = x_vals[3*n//N:4*n//N + 1]
    x_5 = x_vals[4*n//N:5*n//N + 1]
    x_6 = x_vals[5*n//N:6*n//N + 1]
    x_7 = x_vals[6*n//N:7*n//N + 1]
    x_8 = x_vals[7*n//N:]
    
    # Push the x_vals to the engines and execute the trapezoidal rule
    for i, x_vals in enumerate([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]):
        dview.push({"x": x_vals}, targets = i-1)
        dview.execute("value = trap(x)", targets = i-1)

    # Gather the results and sum them
    results = dview.gather("value", block = True)
    client.close()
    return np.sum(results)

if __name__ == "__main__":
    print(prob1())
    dx = {'a':10, 'b':5, 'c':20, 'd':12}
    print('The original dictionary is correctly stored in each engine:', variables(dx))
    client = Client()
    dview = client[:]
    dview.block = True
    for key in dx.keys():
        print('Original:', dx[key])
        print('Pulled:', dview.pull(key)[0])
    client.close()
    means, mins, maxs = prob3()
    print(means, mins, maxs)
    prob4()
    f = lambda x: x**2
    print(parallel_trapezoidal_rule(f, 0, 1, 1000000))

