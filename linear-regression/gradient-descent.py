# Gradient Descent Function
# Here iterations, learning_rate, stopping_threshold
# are hyperparameters that can be tuned

from numpy import *
from random import *

# how many times do we want to train this model
ITERATIONS = 100
STOPPING_THRESHOLD = 1e-6

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate):
    b = starting_b
    m = starting_m

    previous_cost = None
    for i in range(ITERATIONS):

        current_cost = compute_error_for_line_given_points(b, m, points)
        if previous_cost and abs((previous_cost) - current_cost) <= STOPPING_THRESHOLD:
            break
        previous_cost = current_cost
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    
    # Step 1 - collect our data
    # genfromtxt: collect all of the data points from the data file
    # genfromtxt will have two loops: convert each line to string and to appropriate data type
    points = genfromtxt('data.csv', delimiter=',')

    # Step 2 - define our hyperparameters: the parameters that models will use to analyze certain data.


    # how fast (determine the size of step) should our model converge (optimal result, line of best fit)?
    learning_rate = 0.0001

    # slope formula: y = mx + b
    initial_b = 0
    initial_m = 0
    # initial slope: pick random value for the intercept-b and slope-m as an initial guess that gives gradient descent something to improve upon.

    # Step 3 - train our model
    print(f"starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error_for_line_given_points(initial_b, initial_m, points)}")
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate)
    print(f"After {ITERATIONS} iterations b = {b}, m = {m}, error = {compute_error_for_line_given_points(b, m, points)}")


if __name__ == "__main__":
    run()

