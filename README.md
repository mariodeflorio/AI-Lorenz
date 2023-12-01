# AI-Lorenz

Discovering mathematical models that characterize the observed behavior of dynamical systems remains a major challenge, especially for systems in a chaotic regime. The challenge is even greater when the physics underlying such systems is not yet understood, and scientific inquiry must solely rely on empirical data. Driven by the need to fill this gap, we develop a framework that learns mathematical expressions modeling complex dynamical behavior by discovering differential equations from noisy and sparse observable data. We train a small neural network to learn the dynamics of a system and its rate of change in time, which are used as input for a symbolic regression algorithm to autonomously distill the explicit mathematical terms. This, in turn, enables us to predict the future evolution of the dynamical behavior. The performance of this framework is validated by recovering the right-hand sides of certain complex, chaotic systems, such as the well-known Lorenz system, a six-dimensional hyperchaotic system, and the non-autonomous Sprott chaotic system, and comparing them with their known analytical expressions.


## Getting Started

Follow these steps to use the codes and explore the capabilities of the framework.

### Prerequisites

Make sure you have the following prerequisites installed:

- MATLAB
- Python 
- Julia

### Usage (Lorenz System example)

1. Run the file

       ```bash
    RK_lorenz.m
    ```






