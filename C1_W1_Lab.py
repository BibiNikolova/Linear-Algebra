import numpy as np

# I. 1-D arrays

one_dim_arr = np.array([1, 2, 3])
print(one_dim_arr)

b = np.arange(3)
print(b)

c = np.arange(1, 20, 3)
print(c)

evently_spaced_arr = np.linspace(0, 100, 5)
print(evently_spaced_arr)

lin_spaced_arr_int = np.linspace(0, 100, 5, dtype=int)
print(lin_spaced_arr_int)

c_int = np.arange(1, 20, 3, dtype=int)
print(c_int)

b_float = np.arange(3, dtype=float)
print(b_float)

char_arr = np.array(['Welcome to Math for ML!'])
print(char_arr)
print(char_arr.dtype) # Prints the data type of the array

ones_arr = np.ones(3)
print(ones_arr)

zeros_arr = np.zeros(3)
print(zeros_arr)

empty_arr = np.empty(3)
print(empty_arr)

rand_arr = np.random.rand(3)
print(rand_arr)

# II. Multidimensional arrays

two_dim_arr = np.array([[1, 2, 3], [4, 5, 6]])
print(two_dim_arr)


one_dim_arr = np.array([1, 2, 3, 4, 5, 6])
multi_dim_arr = np.reshape(one_dim_arr, (2, 3))
print(multi_dim_arr)

# Finding size, shape and dimension.

print(multi_dim_arr.ndim)

print(multi_dim_arr.shape)

print(multi_dim_arr.size)

# III. Array math operations
print("Array math operations")
arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])
print(arr_1)
print(arr_2)

# Adding two 1-D arrays
print("Adding two 1-D arrays")
addition = arr_1 + arr_2
print(addition)

# Subtracting two 1-D arrays
print("Subtracting two 1-D arrays")
subtraction = arr_1 - arr_2
print(subtraction)

# Multiplying two 1-D arrays elementwise
print("Multiplying two 1-D arrays elementwise")
multiplication = arr_1 * arr_2
print(multiplication)

# Multiplying vector with a scalar (broadcasting)
print("Multiplying vector with a scalar (broadcasting)")
vector = np.array([1, 2])
broadcasting = vector * 1.6
print(broadcasting)

# IV. Indexing and slicing
a = np.array([1, 2, 3, 4, 5])
print("Select the third element of the array " + str(a))
print(a[2])
print("Select the first element of the array " + str(a))
print(a[0])

# Indexing on a 2-D array
two_dim = np.array(([1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]))

# Select element number 8 from the 2-D array using indices i, j
print("Select element number 8 from the 2-D array\n" + str(two_dim))
print(two_dim[2][1]) # two sets of brackets
print(two_dim[2, 1]) # single set of brackets, separated by a comma

# Slicing - (end-exclusive)
print("Slice the array " + str(a) + " from [1] to [4](exclusive)")
sliced_arr = a[1:4]
print(sliced_arr)

print("Slice the array " + str(a) + " to [:3](exclusive)")
sliced_arr = a[:3]
print(sliced_arr)

print("Slice the array " + str(a) + " from [2:]")
sliced_arr = a[2:]
print(sliced_arr)

print("Slice the array " + str(a) + " with step [::2]")
sliced_arr = a[::2]
print(sliced_arr)

print(a == a[::])
print(a == a[:])

# Slice the two_dim array to get the first two rows
sliced_arr_1 = two_dim_arr[0:2]
print(sliced_arr_1)
print("...............")
sliced_two_dim_rows = two_dim_arr[1:3]
print(sliced_two_dim_rows)
print("...............")
sliced_two_dim_cols = two_dim_arr[:,1]
print(sliced_two_dim_cols)

# V. Stacking
a1 = np.array([[1, 1],
               [2, 2]])
a2 = np.array([[3, 3],
              [4, 4]])
print(f'Stacking\na1:\n{a1}')
print(f'a2:\n{a2}')

# Stack the arrays vertically
vert_stack = np.vstack((a1, a2))
print(f'Stack the arrays vertically:\n{vert_stack}')

# Stack the arrays horizontally
horiz_stack = np.hstack((a1, a2))
print(f'Stack the arrays horizontally:\n{horiz_stack}')

# Split the horizontally stacked array into 2 separate arrays of equal size
horiz_split_two = np.hsplit(horiz_stack,2)
print(f'Split the horizontally stacked array into 2 separate arrays of equal size:\n{horiz_split_two}')

# Split the horizontally stacked array into 4 separate arrays of equal size
horiz_split_four = np.hsplit(horiz_stack,4)
print(f'Split the horizontally stacked array into 4 separate arrays of equal size:\n{horiz_split_four}')

# Split the horizontally stacked array after the first column
horiz_split_first = np.hsplit(horiz_stack,[1])
print(f'Split the horizontally stacked array after the first column:\n{horiz_split_first}')

# Split the vertically stacked array into 2 separate arrays of equal size
vert_split_two = np.vsplit(vert_stack,2)
print(f'Split the vertically stacked array into 2 separate arrays of equal size:\n{vert_split_two}')

# Split the vertically stacked array into 4 separate arrays of equal size
vert_split_four = np.vsplit(vert_stack,4)
print(f'Split the vertically stacked array into 4 separate arrays of equal size:\n{vert_split_four}')

# Split the horizontally stacked array after the first and third row
vert_split_first_third = np.vsplit(vert_stack,[1, 3])
print(f'Split the horizontally stacked array after the first and third row:\n{vert_split_first_third}')

