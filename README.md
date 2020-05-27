# simonqiskit
Simons Algorithm for Qiskit

Adaptation of Simons Algorithm for IBM Qiskit / Q. This was taken from one of the original tutorials IBM had fo Qiskit that has since been removed. Original author Rudy Raymond


#### Total population of bitstrings (including all 0's)
[(2^n)-1]*[(2^n)!]/[(2^(n-1))!]

#### Sample Size Formula 
s = n*x / (x + n - 1)
x = ((1.96)*(1.96))*0.50*0.50 / (0.05)*0.05) = 385

s = (36 * 385) / (385 + 36 - 1) = 35
s = (11760 * 385) / (385 + 11760 - 1) = 372
s = (7783776000 * 385) / (385 + 7783776000 -1) = 384
s = (3.89e23 * 385) / (385 + 3.89e23 - 1) = 385
s = (3.037e55 * 385) / (385 + 3.037e55 - 1) = 385
s = (3.85e128 * 385) / (385 + 3.85e128 - 1) = 385


