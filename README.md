# STL-Constrained Zonotopic Predictive Control (STL-ZPC)

## Overview

This repository provides a framework for **Zonotopic Predictive Control (ZPC)** under **Signal Temporal Logic (STL) constraints** to ensure safe trajectory planning for autonomous vehicles.  

It integrates:  
- **Data-Driven Reachability Analysis** to approximate how vehicle states evolve over time using learned models.  
- **STL Specification Handling** to encode temporal safety requirements (e.g., speed limits, lane boundaries).  
- **Nonlinear Vehicle Models** for realistic simulation and control.  
- **Trajectory Optimization** using advanced constrained optimization techniques.  
- **Visualization Tools** to plot reachable sets, vehicle trajectories, and controller performance.  

The framework supports:  
- Training and evaluating data-driven vehicle models.  
- Synthesizing control inputs that ensure compliance with STL constraints.  
- Handling failure and emergency braking scenarios to guarantee functional safety.

This package is aimed at autonomous driving applications, enabling runtime functional safety and trajectory planning within constrained environments.

## Usage

The repository provides two primary entry points for different tasks:

### Training a Data-Driven Vehicle Model

This trining part is based on the work published by Mahmoud Selim et al. [Safe Reinforcement Learning Using Black-Box Reachability Analysis](https://ieeexplore.ieee.org/document/9833266).

Use the `train_data_driven_model.py` script to train a model for data-driven reachability analysis:

```bash
python3 -m stl_constrained_zpc.scripts.vehicle_model.data_driven.train_data_driven_model --batch_name roundabout --frequency 10 --steps 2 -p --initpoints 1000
```

The trained model will be stored and later used in predictive control and reachability computations.

In this example, we provide a small amount of data about (X-, X+, U) of a real-world autonomous vehicle driving in a roundabout.

<p align="center">
  <img src="https://raw.githubusercontent.com/cconejob/stl_constrained_zpc/master/stl_constrained_zpc/plots/example/roundabout_data.png" alt="Training data example">
</p>

### Running STL-Constrained ZPC

Use the main script `stl_constrained_zpc.py` to execute the controller:

```bash
python3 -m stl_constrained_zpc.scripts.stl_constrained_zpc -s 5 --model_name roundabout_1000_2 -r 2 -sw 100 -p -i 335 -fi 10 --yaw
```

This runs the predictive controller with STL specifications and saves the results in the plots/ folder.

<p align="center">
  <img src="https://raw.githubusercontent.com/cconejob/stl_constrained_zpc/master/stl_constrained_zpc/plots/example/safe_stop.gif" alt="Data-driven Zonotopic Predictive Control with STL constraints">
</p>

## Related Research

- Integrating Signal Temporal Logic into Zonotopic Predictive Control for Autonomous Vehicles (Conference Article, IEEE ITSC 2025)

## Contact Information and Acknowledgement

For further information regarding this project, please feel free to reach out to Carlos Conejo [carlos.conejo@upc.edu](mailto:carlos.conejo@upc.edu).

This project was mainly developed at the [Professorship for Cyber Physical Systems in Heilbronn](https://www.ce.cit.tum.de/en/ce/research/professorships/hncps/) in collaboration with the [Institut de Robòtica i Informàtica Industrial (IRI)](https://www.iri.upc.edu/), a joint university research center of the Polytechnic University of Catalonia (UPC) and the Spanish National Research Council (CSIC). 

Research partially funded by the Spanish State Research Agency (AEI), the European Regional Development Fund (ERFD) through the SaCoAV project (ref. PID2020-114244RB-I00), and the iMOVE 2024 grant (ref. 24281). Also funded by Renault Group through the Industrial Doctorate "Safety of Autonomous Vehicles" (ref. C12507).