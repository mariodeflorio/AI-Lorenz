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
y4 = []
y5 = []  
y6 = []  
f = []   # Target variable 'f'
g = []   # Target variable 'g'
h = []   # Target variable 'h'
i = []   # Target variable 'i'
j = []   # Target variable 'j'
k = []   # Target variable 'k'
t = []

with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      t.append(float(row[0]))
      y1.append(float(row[1]))
      y2.append(float(row[2]))
      y3.append(float(row[3]))
      y4.append(float(row[4]))
      y5.append(float(row[5])) 
      y6.append(float(row[6])) 
      f.append(float(row[7]))
      g.append(float(row[8]))
      h.append(float(row[9]))
      i.append(float(row[10]))  
      j.append(float(row[11]))  
      k.append(float(row[12]))  
      line_count += 1
    print(f'Processed {line_count} lines.')

t = np.array(t).reshape(-1, 1)
y1 = np.array(y1).reshape(-1, 1)
y2 = np.array(y2).reshape(-1, 1)
y3 = np.array(y3).reshape(-1, 1)
y4 = np.array(y4).reshape(-1, 1)
y5 = np.array(y5).reshape(-1, 1)
y6 = np.array(y6).reshape(-1, 1)
f = np.array(f).reshape(-1, 1)
g = np.array(g).reshape(-1, 1)
h = np.array(h).reshape(-1, 1)
i = np.array(i).reshape(-1, 1)
j = np.array(j).reshape(-1, 1)
k = np.array(k).reshape(-1, 1)

d = np.concatenate((t, y1, y2, y3, y4, y5, y6, f, g, h, i, j, k), axis=1)  # Include all variables as columns

X = d[:, 1:7]  # Input features: 'y1,' 'y2,' 'y3', y4 ,y5, y6


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
model_g = PySRRegressor(
    niterations=30,
    binary_operators=["+", "*","-"],
    **pysr_params
)

# Symbolic Regression for 'h'
model_h = PySRRegressor(
    niterations=30,
    binary_operators=["+", "*","-"],
    **pysr_params
)

# Symbolic Regression for 'i'
model_i = PySRRegressor(
    niterations=30,
    binary_operators=["+", "*","-"],
    **pysr_params
)

# Symbolic Regression for 'j'
model_j = PySRRegressor(
    niterations=30,
    binary_operators=["+", "*","-"],
    **pysr_params
)

# Symbolic Regression for 'k'
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

model_g.fit(X, g)
print(f"Model for 'g': {model_g}")
print(f"Model SymPy for 'g': {model_g.sympy()}")
print(f"Latex Equation for 'g': {model_g.latex()}")

model_h.fit(X, h)
print(f"Model for 'h': {model_h}")
print(f"Model SymPy for 'h': {model_h.sympy()}")
print(f"Latex Equation for 'h': {model_h.latex()}")

model_i.fit(X, i)
print(f"Model for 'i': {model_i}")
print(f"Model SymPy for 'i': {model_i.sympy()}")
print(f"Latex Equation for 'i': {model_i.latex()}")

model_j.fit(X, j)
print(f"Model for 'j': {model_j}")
print(f"SymPy Expression for 'j': {model_j.sympy()}")
print(f"Latex Equation for 'j': {model_j.latex()}")

model_k.fit(X, k)
print(f"Model for 'k': {model_k}")
print(f"SymPy Expression for 'k': {model_k.sympy()}")
print(f"Latex Equation for 'k': {model_k.latex()}")




print()
print(f"Model SymPy for 'f': {model_f.sympy()}")
print(f"Model SymPy for 'g': {model_g.sympy()}")
print(f"Model SymPy for 'h': {model_h.sympy()}")
print(f"Model SymPy for 'i': {model_i.sympy()}")
print(f"Model SymPy for 'j': {model_j.sympy()}")
print(f"Model SymPy for 'k': {model_k.sympy()}")

