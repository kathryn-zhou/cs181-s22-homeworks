#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import math
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false

def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
    
    res = xx
    if part == "a":
        for i in range(5):
            res = np.vstack([res, xx**(i+1)])
    elif part == "b":
        for u in range(1960, 2015, 5):
            res = np.vstack([res, np.exp(-(xx-u)**2/25)])
    elif part == "c":
        for j in range(5):
            res = np.vstack([res, np.cos(xx / (j+1))])
    elif part == "d":
        for j in range(25):
            res = np.vstack([res, np.cos(xx / (j+1))])
        
    res[0] = np.array([1 for _ in range(len(res[0]))])
        
    return res.T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# 4.1
for part in ["a", "b", "c", "d"]:
    # Compute the regression line on a grid of inputs.
    # DO NOT CHANGE grid_years!!!!!
    grid_years = np.linspace(1960, 2005, 200)
    X = make_basis(years, part=part)
    w = find_weights(X, republican_counts)

    train_X = make_basis(grid_years, part=part)
    grid_Yhat  = np.dot(train_X, w)
    # TODO: plot and report sum of squared error for each basis
    Y = np.dot(X, w)
    print(f"Residual sum of squares for 4.1{part}:", sum((Y-republican_counts)**2))


    # Plot the data and the regression line.
    plt.title(f"Regression using basis ({part})")
    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.savefig("4.1 basis" + str(part) + ".png")
    plt.show()

# 4.2
for part in ["a", "c", "d"]:
    X = make_basis(sunspot_counts[years<last_year], part=part, is_years=False)
    w = find_weights(X, republican_counts[years<last_year])

    grid_sunspots = np.linspace(min(sunspot_counts[years<last_year]), max(sunspot_counts[years<last_year]),200)
    train_X = make_basis(grid_sunspots, part=part, is_years=False)
    grid_Yhat  = np.dot(train_X, w)
    # # TODO: plot and report sum of squared error for each basis
    Y = np.dot(X, w)
    print(f"Residual sum of squares for 4.2{part}:", sum((Y-republican_counts[years<last_year])**2))


    # # Plot the data and the regression line.
    plt.title(f"Regression using basis ({part})")
    plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("Sunspot Counts")
    plt.ylabel("Number of Republicans in Congress")
    plt.savefig("4.2 basis" + str(part) + ".png")
    plt.show()