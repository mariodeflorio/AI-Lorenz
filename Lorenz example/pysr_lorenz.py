import numpy as np
from pysr import PySRRegressor
import csv
import os

directory = 'data'  # Replace 'data' with your actual directory name

file_path = os.path.join(directory, 'pysr_data.csv')

## Data Import
y1 = []
y2 = []
y3 = []
f = []  # Target variable 'f'
h = []  # Target variable 'h'
k = []  # Target variable 'k'
t = []

# with open('t_y1_y2_y3_f_g_h.csv') as csv_file:
with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      t.append(float(row[0]))
      y1.append(float(row[1]))
      y2.append(float(row[2]))
      y3.append(float(row[3]))
      f.append(float(row[4]))
      h.append(float(row[5]))
      k.append(float(row[6]))
      line_count += 1
    print(f'Processed {line_count} lines.')

t = np.array(t).reshape(-1, 1)
y1 = np.array(y1).reshape(-1, 1)
y2 = np.array(y2).reshape(-1, 1)
y3 = np.array(y3).reshape(-1, 1)
f = np.array(f).reshape(-1, 1)
h = np.array(h).reshape(-1, 1)
k = np.array(k).reshape(-1, 1)

d = np.concatenate((t, y1, y2, y3, f, h, k), axis=1)  # Include 'y1,' 'y2,' 'y3,' 'g,' 'f,' and 'h' as columns

X = d[:, 1:4]  # Input features: 'y1,' 'y2,' 'y3'


pysr_params = dict(
    populations=30,
    model_selection="best",
)


# Symbolic Regression for 'f'
model_f = PySRRegressor(
    niterations=30,
    binary_operators=["+", "*","-"],
    **pysr_params
)

# Symbolic Regression for 'g'
model_h = PySRRegressor(
    niterations=30,
    binary_operators=["+", "*","-"],
    **pysr_params
)

# Symbolic Regression for 'h'
model_k = PySRRegressor(
    niterations=30,
    binary_operators=["+", "*","-"],
    **pysr_params
)

# Run model:

model_f.fit(X, f)

print(f"Model for 'f': {model_f}")
print(f"Model SymPy for 'f': {model_f.sympy()}")
print(f"Latex Equation for 'f': {model_f.latex()}")





model_h.fit(X, h)

print(f"Model for 'h': {model_h}")
print(f"Model SymPy for 'h': {model_h.sympy()}")
print(f"Latex Equation for 'h': {model_h.latex()}")





model_k.fit(X, k)

print(f"Model for 'k': {model_k}")
print(f"Model SymPy for 'k': {model_k.sympy()}")
print(f"Latex Equation for 'k': {model_k.latex()}")




print()
print()
print(f"Model SymPy for 'f': {model_f.sympy()}")
print(f"Model SymPy for 'h': {model_h.sympy()}")
print(f"Model SymPy for 'k': {model_k.sympy()}")





