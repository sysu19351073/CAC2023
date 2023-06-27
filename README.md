# CAC2023
## Introduction
This project is a reference for CAC2023 submissions.

## Code and Model
The 4 folders contain Arduino, Python, MATLAB code, STL, and CAD models.

### Arduino
Arduino code needs to be used with Python code, and after burning to Arduino Uno, connect Arduino Uno to the computer.
The Arduino contains the operating code of the stepper motor and servo, and can control the cooperative mechanical gripper operation after receiving instructions from the computer.

### CAD
The CAD folder contains CAD and STL models. STL models can be used for 3D printing assembly prototype platforms.

### Python
The Cooperative mechanical gripper folder is the Python code.
Arduino is the main program, and the generation of state diagrams is implemented in the solve.

### MATLAB
This folder includes the MATLAB/Simulink simulator, which requires the use of models in CAD.
