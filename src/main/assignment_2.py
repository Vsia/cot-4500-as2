#Vallesia Pierre Louis
#Assignment 2
#cot4500
import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)
#from scipy.interpolate import CubicSpline


#Number 1 - Using Neville’s method, find the 2nd degree interpolating value for f(3.7) for the following  set of data
def question1_nevilles_method(x_points, y_points, x):

  num_of_points = len(x_points)
  matrix = np.zeros((3, 3))

  for counter, row in enumerate(matrix):
    row[0] = y_points[counter]
  num_of_points = len(x_points)
  for i in range(1, num_of_points):
    for j in range(1, i + 1):
      first_multiplication = (x - x_points[i - j]) * matrix[i][j - 1]
      second_multiplication = (x - x_points[i]) * matrix[i - 1][j - 1]

      denominator = x_points[i] - x_points[i - j]

      coefficient = (first_multiplication -
                     second_multiplication) / denominator
      matrix[i][j] = coefficient

  print(matrix[i][j], "\n")


if __name__ == "__main__":
  # point setup
  x_points = [3.6, 3.8, 3.9]
  y_points = [1.675, 1.436, 1.318]
  approximating_value = 3.7
  question1_nevilles_method(x_points, y_points, approximating_value)


#-------------------------------------------------------------------------------
#number 2 - Using Newton’s forward method, print out the polynomial approximations for degrees 1, 2, and 3 using the following set of data
# part a- creatre the table first
def divided_difference_table(x_points, y_points):
  # set up the matrix
  size: int = len(x_points)
  matrix: np.array = np.zeros((size, size))
  # fill the matrix
  for index, row in enumerate(matrix):
    row[0] = y_points[index]
  # populate the matrix (end points are based on matrix size and max operations we're using)
  for i in range(1, size):
    for j in range(1, i + 1):
      # the numerator are the immediate left and diagonal left indices...
      numerator = matrix[i][j - 1] - matrix[i - 1][j - 1]
      # the denominator is the X-SPAN...
      denominator = (x_points[i]) - (x_points[i - j])
      operation = numerator / denominator
      # cut it off to view it more simpler
      matrix[i][j] = format(operation)
      #printing out the approxiations
      if i == j:
        print(matrix[i][j])

  return matrix


def get_approximate_result(matrix, x_points, value):
  # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x
  reoccuring_x_span = 1
  reoccuring_px_result = matrix[0][0]

  # we only need the diagonals...and that starts at the first row...
  for index in range(1, 4):
    polynomial_coefficient = matrix[index][index]
    # we use the previous index for x_points....
    reoccuring_x_span *= (value - x_points[index - 1])
    # get a_of_x * the x_span
    mult_operation = polynomial_coefficient * reoccuring_x_span
    # add the reoccuring px result
    reoccuring_px_result += mult_operation

  print(reoccuring_px_result,"\n")


if __name__ == "__main__":
  # point setup
  x_points = [7.2, 7.4, 7.5, 7.6]
  y_points = [23.5492, 25.3913, 26.8224, 27.4589]
  divided = divided_difference_table(x_points, y_points)

#-------------------------------------------------------------------------------
#Number 3 - Using the results from 3, approximate f(7.3)?
question3_approximation = get_approximate_result(divided, x_points, 7.3)

#------------------------------------------------------------------------------
#number 4- Using the divided difference method, print out the Hermite polynomial approximation matrix
def apply_div_dif(matrix: np.array):
  size = len(matrix)
  
  for i in range(2, size):
    for j in range(2, i + 2):
      # skip if value is prefilled (we dont want to accidentally recalculate...)
      if j >= len(matrix[i]) or matrix[i][j] != 0:
        continue

      # get left cell entry
      left: float = matrix[i][j - 1]
      # get diagonal left entry
      diagonal_left: float = matrix[i - 1][j - 1]
      # order of numerator is SPECIFIC.
      numerator: float = left - diagonal_left
      # denominator is current i's x_val minus the starting i's x_val....
      denominator = matrix[i][0] - matrix[i - (j + 1)][0]
      # something save into matrix
      operation = numerator / denominator
      matrix[i][j] = operation
  print(matrix)
  return matrix

def hermite_interpolation():
  x_points = [3.6, 3.8, 3.9]
  y_points = [1.675, 1.436, 1.318]
  slopes = [-1.195, -1.188, -1.182]

  # matrix size changes because of "doubling" up info for hermite
  num_of_points = len(x_points)
  twice_size = (2 * num_of_points)
  matrix = np.zeros((twice_size, twice_size))
  # populate x values (make sure to fill every TWO rows)
  for x in range(num_of_points):
    matrix = [2 * x][0] = x_points[x]
    matrix = [2 * x + 1][0] = x_points[x]
    break

  # prepopulate y values (make sure to fill every TWO rows)
  for x in range(num_of_points):
    matrix = [2 * x][1] = y_points[x]
    matrix[2 * x + 1][1] = y_points[x]
    break
  # prepopulate with derivates (make sure to fill every TWO rows. starting row CHANGES.)
  for x in range(num_of_points):
    matrix = [2 * x + 1][2] = slopes[x]
    break
    filled_matrix = apply_div_dif(matrix)
  print(filled_matrix)
#-------------------------------------------------------------------------------

#number 5- Using cubic spline interpolation, solve for the following using this set of data:

def cubic_spline(x_points, y_points):

  num_of_points = len(x_points)
  matrix = np.zeros((num_of_points, num_of_points))

  matrix[0, 0] = 1
  matrix[num_of_points - 1, num_of_points - 1] = 1


  for i in range(1, num_of_points - 1):
    matrix[i, i - 1] = x_points[i] - x_points[i - 1]
    matrix[i, i] = 2 * (x_points[i + 1] - x_points[i - 1])
    matrix[i, i + 1] = x_points[i + 1] - x_points[i]
  print(matrix,"\n")

# print vector b
  vector_b = np.zeros(num_of_points)
  for i in range(1, num_of_points - 1):
    num=3
    vector_b[i] = num * (y_points[i+1] - y_points[i]) / (x_points[i+1] - x_points[i]) - \
           num * (y_points[i] - y_points[i-1]) / (x_points[i] - x_points[i-1])
  print(vector_b,"\n")
  
# print  vector x
  vector_x = np.linalg.solve(matrix, vector_b)
  print(vector_x,"\n")

if __name__ == "__main__":
  x_points = np.array([2, 5, 8, 10])
  y_points = np.array([3, 5, 7, 9])
  cubic_spline(x_points, y_points)
