# simonqiskit
Simons Algorithm for Qiskit

Adaptation of Simons Algorithm for IBM Qiskit / Q. This was taken from one of the original tutorials IBM had fo Qiskit that has since been removed. Original author Rudy Raymond


# Total population of bitstrings (including all 0's)
[(2^n)-1]*[(2^n)!]/[(2^(n-1))!]

# n = 3
(2^3) - 1 * (2^3)! / 2^(3-1)!
7 * 40320 / 24 = 11760

# Assumptions:
1. You seek a confidence level of 95% (5% error), so Z = 1.96
2. You assume your population is made up of 50% correct and 50% incorrect, then p=.50 & q=(1-p) = .50
3. You look for 5% desired level of precision, so e = 0.05
The formula for sample size is: n = [Z*Z*p*q] / [e*e] =
(1.96)(1.96)(0.50)(0.50) / (0.05)(0.05) = 384.16


