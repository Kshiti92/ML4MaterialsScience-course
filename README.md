### ML for Materials Science: Predicting Thermal Conductivity in SiOC Glass Ceramics
Recording learnings from the course "Machine Learning for Materials Science" by TU Darmstadt.
A hands-on project applying CNNs and Bayesian optimization to predict and maximise effective thermal conductivity in SiOC glass ceramic nanocomposites — demonstrating a complete structure-property discovery workflow.

#### Project Overview
The material system studied is SiOC glass ceramics — a nanocomposite with three interacting phases whose composition directly controls its effective thermal conductivity. SiOC is of significant engineering interest due to its high chemical durability, excellent resistance to oxidative and corrosive environments, and high thermal stability.
The project is structured as two connected tasks:
##### Task 1 — CNN for Structure-Property Prediction:
360 computer-generated 3D nanocomposite microstructures (each a 100×100×100 .npy file) were used alongside thermal conductivity values from finite element simulations. After exploratory data analysis — including volume fraction calculation, correlation analysis, and benchmarking with linear regression and a simple neural network — a Convolutional Neural Network (CNN) was trained to learn the relationship between microstructure images and effective thermal conductivity.
##### Task 2 — Bayesian Optimization for Composition Design:
Using the structure-property relationship learned by the CNN, Bayesian optimization was applied to search the composition space and identify the composition that maximises thermal conductivity — without needing to run a physical or simulated experiment for every candidate.
These two tasks form a closed loop: the CNN learns the map, Bayesian optimization exploits it.