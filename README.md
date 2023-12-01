# AI-Lorenz

Discovering mathematical models that characterize the observed behavior of dynamical systems remains a major challenge, especially for systems in a chaotic regime. The challenge is even greater when the physics underlying such systems is not yet understood, and scientific inquiry must solely rely on empirical data. Driven by the need to fill this gap, we develop a framework that learns mathematical expressions modeling complex dynamical behavior by discovering differential equations from noisy and sparse observable data. We train a small neural network to learn the dynamics of a system and its rate of change in time, which are used as input for a symbolic regression algorithm to autonomously distill the explicit mathematical terms. This, in turn, enables us to predict the future evolution of the dynamical behavior. The performance of this framework is validated by recovering the right-hand sides of certain complex, chaotic systems, such as the well-known Lorenz system, a six-dimensional hyperchaotic system, and the non-autonomous Sprott chaotic system, and comparing them with their known analytical expressions.

<img src="https://github.com/mariodeflorio/AI-Lorenz/blob/main/AI_lorenz_scheme.png">



## Getting Started

Follow these steps to use the codes and explore the capabilities of *AI-Lorenz*.

### Prerequisites

Make sure you have the following prerequisites installed:

* MATLAB
* Python 
* [Julia](https://julialang.org/downloads/)

### Usage (Lorenz System example)

1. Run the file 'RK_lorenz.m' for data generation. 

   This MATLAB script generates data for the Lorenz system using the Runge-Kutta method. The user can define the time domain, the step size 'h_step', governing parameters, and initial conditions. The script saves the results in a file named 'data_generated.mat' into the directory 'data'. The generated data includes solutions for the variables y1, y2, and y3, as well as the right-hand sides (RHS) of the Lorenz system. 
   
2. Run the file 'bbxtfc_lorenz.m' for dynamics learning and RHS extraction. 

   This MATLAB script performs the X-TFC algorithm with domain decomposition for black-box learning of the Lorenz system. The data generated in step 1. is loaded, and the user can add noise to it by modifying the variable *noise_std*. Depending on the presence or absence of noise in the data, different values of collocation points per each sub-domain *N*, number of neurons *m*, and time step length *t_step* can be chosen. Follows the list of tunable parameters:
      * *N*, number of collocation points per each sub-domain
      * *m*, number of neurons
      * *t_step*, length of sub-domains
      * *LB*, Lower boundary for weight and bias samplings
      * *UB*, Upper boundary for weight and bias samplings
      * *IterMax*, maximum number of iterations of the least-squares algorithm 
      * *IterTol*, tolerance of the least-squares algorithm
      * *type_act*, select the activation function to use.
        
   If the data is noisy, the learned dynamics and RHS can present outliers. Smooth them with a Savitzky–Golay filter
 and tune the following parameters for the :
      * *window_size*, Frame length, specified as a positive odd integer
      * *polynomial_order*, Polynomial order, specified as a positive integer (must be smaller than window_size).

   The script prints the mean absolute errors for learned dynamics and RHS, and the variables are saved in 'pysr_data.csv' (to be used in step 3.) and in 'bbxtfc_data.mat' (to be used in step 4.).

3. Run the file 'pysr_lorenz.py' for symbolic regression.

   This Python script performs symbolic regression with [PySR](https://github.com/MilesCranmer/PySR) algorithm to distill the mathematical expressions that best fit the provided input data. The learned dynamics and RHS from step 2. are loaded, and the user can modify (for each state variable of the system), the following parameters: 
      * *population*, Number of populations running
      * *niterations*, Number of iterations of the algorithm to run. The best equations are printed and migrate between populations at the end of each iteration
      * *binary_operators*, List of strings for binary operators used in the search
      * *unary_operators*, Operators which only take a single scalar as input. For example, "cos" or "exp".

   The script will print the best candidate mathematical expressions that are used to build the discovered dynamical system.

4. Run the file 'SR_test_lorenz.m' for prediction and validation.

   This MATLAB script solves the system on ODEs discovered by PYSR in step 3. and compares the discovered chaotic trajectory with the synthetic trajectory generated in step 1. All the needed variables are loaded from the 'data' directory. The user only needs to build the system of ODEs discovered with symbolic regression.  








