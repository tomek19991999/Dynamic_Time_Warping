import matplotlib.pylab as plt
import numpy as np
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Calculate distance matrix. d(x_i,y_i)=|x_i-y_j|
def compute_distance_matrix(dist_matrix):
    i = 0
    while i < N:
        j = 0
        while j < M:
            dist_matrix[i, j] = abs(x[i] - y[j])
            j += 1
        i += 1
    return dist_matrix


# Calculate cost matrix. γ(i,j)=d(x_i,y_i)+min[γ(i-1,j-1),γ(i-1,j),γ(i,j-1)]
def calculate_cost_matrix(dist_matrix):
# Initialize cost matrix with zeros
    dim_N, dim_M = dist_matrix.shape
    cost_mat = np.zeros((dim_N + 1, dim_M + 1))

    # Add extra row and column - they're needed for finding min() in our matrix (near edges).
    i = 1
    while i <= dim_N:
        cost_mat[i, 0] = np.inf
        i += 1
    
    i = 1
    while i <= dim_M:
        cost_mat[0, i] = np.inf
        i += 1

    # Creating supplementary path matrix (will be utilized in the following step, for computing optimal path)
    path_mat = np.zeros((dim_N, dim_M))

    # Construct cost matrix. γ(i,j)=d(x_i,y_i)+min[γ(i-1,j-1),γ(i-1,j),γ(i,j-1)]
    i = 1
    while i <= dim_N:
        j = 1
        while j <= dim_M:
            # Generate list with permitted cells
            permitted_cells = [
                cost_mat[i - 1, j - 1],  # index 0
                cost_mat[i - 1, j],  # index 1
                cost_mat[i, j - 1],  # index 2
            ]
            # Obtain index of cell with min value and calculate.
            min_index = np.argmin(permitted_cells)
            cost_mat[i, j] = (
                dist_matrix[i - 1, j - 1] + permitted_cells[min_index]
            )  # γ(i,j)=d(x_i,y_i)+min[γ(i-1,j-1),γ(i-1,j),γ(i,j-1)]
            # Take min value to additional matrix with min values
            path_mat[i - 1, j - 1] = min_index
            j += 1
        i += 1

    # Remove infinity values and compute best path
    cost_mat = cost_mat[1:, 1:]
    optimal_path = calculate_path(path_mat)
    return (optimal_path, cost_mat)


def calculate_path(path_mat):
# Get the shape of the matrix and initialize the end coordinates
    dim_N, dim_M = path_mat.shape
    end_coord_n, end_coord_m = dim_N - 1, dim_M - 1

    # Create a list to hold the best path coordinates
    best_path = [(end_coord_n, end_coord_m)]

    # Compute the best path
    while end_coord_n > 0 or end_coord_m > 0:
        # Choose the path based on the current cell value
        chosen_path = path_mat[end_coord_n, end_coord_m]
        if chosen_path == 0:  # Diagonal movement
            end_coord_n -= 1
            end_coord_m -= 1
        elif chosen_path == 1:  # Vertical movement
            end_coord_n -= 1
        else:  # Horizontal movement
            end_coord_m -= 1
        # Append the new coordinates to the best path
        best_path.append((end_coord_n, end_coord_m))

    # Reverse the order of the path coordinates to get the correct sequence
    best_path = best_path[::-1]

    return best_path

def plot_distance_matrix(distance_matrix, x, y, best_path_x, best_path_y):
    fig, plot = plt.subplots(1, 1, figsize=(12, 9)) # create window with 2 plots 

    scale = plot.imshow(
     distance_matrix, cmap=plt.cm.binary, interpolation="nearest", origin="lower"
    ) # plot the distance matrix
    plot.set_title("Distance matrix [d(x_i,y_i)=|x_i-y_j|]", size=18) # set title to plot
    plot.plot(best_path_y, best_path_x) # draw best path on distance matrix
    fig.colorbar(scale, ax=plot) # plot a colorbar with a graduated scale

    divider = make_axes_locatable(plot) # create local plots
    trace_x = divider.append_axes("left", 1, pad=0.5, sharey=plot) # create space for x trace plot
    trace_x.plot(x, np.arange(x.shape[0])) # create x trace
    trace_x.xaxis.set_tick_params(labelbottom=False) # remove x axis values
    trace_x.yaxis.set_tick_params(labelleft=False) # remove y axis values

    trace_y = divider.append_axes("bottom", 1, pad=0.5, sharex=plot) # create space for y trace plot
    trace_y.plot(np.arange(y.shape[0]), y) # create y trace
    trace_y.xaxis.set_tick_params(labelbottom=False) # remove x axis values
    trace_y.yaxis.set_tick_params(labelleft=False) # remove y axis values

def plot_cost_matrix(cost_matrix, x, y, best_path_x, best_path_y):
    fig, plot = plt.subplots(1, 1, figsize=(12, 9)) # create window with 2 plots

    scale = plot.imshow(
     cost_matrix, cmap=plt.cm.binary, interpolation="nearest", origin="lower"
    ) # plot the cost matrix
    plot.set_title("Cost matrix [γ(i,j)=d(x_i,y_i)+min[γ(i-1,j-1),γ(i-1,j),γ(i,j-1)]]", size=18) # set title to plot
    plot.plot(best_path_y, best_path_x) # draw best path on cost matrix
    fig.colorbar(scale, ax=plot) # plot a colorbar with a graduated scale

    divider = make_axes_locatable(plot) # create local plots
    trace_x = divider.append_axes("left", 1, pad=0.5, sharey=plot) # create space for x trace plot
    trace_x.plot(x, np.arange(x.shape[0])) # create x trace
    trace_x.xaxis.set_tick_params(labelbottom=False) # remove x axis values
    trace_x.yaxis.set_tick_params(labelleft=False) # remove y axis values

    trace_y = divider.append_axes("bottom", 1, pad=0.5, sharex=plot) # create space for y trace plot
    trace_y.plot(np.arange(y.shape[0]), y) # create y trace
    trace_y.xaxis.set_tick_params(labelbottom=False) # remove x axis values
    trace_y.yaxis.set_tick_params(labelleft=False) # remove y axis values
    plt.show()

# Sinus and second sinus, but przesuniety
t = np.arange(0, 2 * np.pi, 0.05)  # step 0.2, from 0 to 2*pi
y1 = np.sin(t)
y2 = np.sin(t - np.pi / 2)

x = np.array(y1)
y = np.array(y2)


"""
plt.plot(t, x, label="sin(x)")
plt.plot(t, y, label="sin(x - pi/2)")
plt.legend()
plt.show()
"""

# take amount of steps, make matrix and fill cells with 0
N = x.shape[0]
M = y.shape[0]
matrix_with_zeros = np.zeros((N, M))

# STEP1 - Calculate distance matrix. d(x_i,y_i)=|x_i-y_j|
distance_matrix = compute_distance_matrix(matrix_with_zeros)

# STEP2 - Calculate cost matrix and best path.
best_path, cost_matrix = calculate_cost_matrix(distance_matrix)
best_path_x, best_path_y = zip(*best_path)  # getting x and y components of the coordinates

# STEP3 - calculate distance between two plots
distance = (abs(min(x)) + abs(max(y))) / 2 + 0.5 #abs(1)= |1|


# STEP4 - make plots
# two plots, moved
plt.figure(figsize=(10, 7))
plt.plot(np.arange(x.shape[0]), x + distance, c="#00ff00", label="x")
plt.plot(np.arange(y.shape[0]), y - distance, c="#0000ff", label="y")
plt.legend()
plt.axis("off")
#plt.show()

# two plots, on each other
plt.figure(figsize=(10, 7))
plt.plot(np.arange(x.shape[0]), x, c="#00ff00", label="x")
plt.plot(np.arange(y.shape[0]), y, c="#0000ff", label="y")
plt.legend()
# plt.show()

# two plots, on each other
plt.figure(figsize=(10, 7))
plt.plot(np.arange(x.shape[0]), x + distance, c="#00ff00", label="x")
plt.plot(np.arange(y.shape[0]), y - distance, c="#0000ff", label="y")
for x_i, y_j in best_path:
    plt.plot([x_i, y_j], [x[x_i] + distance, y[y_j] - distance], c="#808080")  # make lines connected to points
plt.legend()
plt.axis("off")
#plt.show()

# STEP6 - plot distance matrix
plot_distance_matrix(distance_matrix, x, y, best_path_x, best_path_y)

# STEP7 - plot cost matrix
plot_cost_matrix(cost_matrix, x, y, best_path_x, best_path_y)
