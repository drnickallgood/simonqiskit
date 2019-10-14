import sys
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute,Aer 
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.visualization import circuit_drawer
from sympy import Matrix, pprint, MatrixSymbol, expand, mod_inverse

# 2 qubits
# blackbox returns
# 00 -> 00
# 01 -> 00
# 10 -> 11
# 11 -> 11
###
#
# 00 xor f(00) = 00
# 01 xor f(01) = 01
# 10 xor f(10) = 01
# 11 xor f(11) = 00
#

# hidden string
s = "010101"
n = len(s)


# Create registers

# 2^n quantum registers half for control, half for data, 
# n classical registers for the output
qr = QuantumRegister(2*n)
cr = ClassicalRegister(n)

circuitName = "Simon"
simonCircuit = QuantumCircuit(qr,cr)

# Apply hadamards prior to oracle 
for i in range(n):
	simonCircuit.h(qr[i])
	
# Barrier
simonCircuit.barrier()


#### Blackbox Function #####
# QP's don't care about this, we do#
#############################

# Copy first register to second by using CNOT gates
for i in range(n):
	simonCircuit.cx(qr[i],qr[n+i])
	
# get the small index j such it's "1"
j = -1
#reverse the string so that it takes
s = s[::-1]
for i, c in enumerate(s):
	if c == "1":
		j = i
		break
		
# 1-1 and 2-1 mapping with jth qubit 
# x is control to xor 2nd qubit with a
for i, c in enumerate(s):
    if c == "1" and j >= 0:
        #simonCircuit.x(qr[j])
        simonCircuit.cx(qr[j], qr[n+i]) #the i-th qubit is flipped if s_i is 1
        #simonCircuit.x(qr[j])

# Random peemutation
# This part is how we can get by with 1 query of the oracle and better
# simulates quantum behavior we'd expect
perm = list(np.random.permutation(n))

# init position
init = list(range(n))

i = 0

while i < n:
	if init[i] != perm[i]:
		k = perm.index(init[i])
		simonCircuit.swap(qr[n+i],qr[n+k])  #swap gate on qubits
		init[i], init[k] = init[k], init[i] # mark the swapped qubits
	else:
		i += 1
		
# Randomly flip qubit
for i in range(n):
	if np.random.random() > 0.5:
		simonCircuit.x(qr[n+i])

simonCircuit.barrier() 

### END OF BLACKBOX FUNCTION 

# Apply hadamard gates to registers again
for i in range(n):
	simonCircuit.h(qr[i])
		

	
# draw circuit
#circuit_drawer(simonCircuit)


### simulation ###
simonCircuit.barrier(qr)
simonCircuit.measure(qr[0:n],cr)
backend = Aer.get_backend('qasm_simulator') 
print(simonCircuit)
shots = 100000
job = execute(simonCircuit, backend, shots=shots)

result = job.result()

counts = result.get_counts(simonCircuit)

#plot_histogram(counts)
#print(counts)


print("\nSimulation: Resulting Values and Probabilities")
print("===============================================\n")
print("Simulated Runs:",shots,"\n")

for key, val in counts.items():
        prob = val / shots
        print("Period:", key, ", Counts:", val, ", Probability:", prob)

print("")

# Classical post processing via Guassian elimination for the linear equations
# Y a = 0
# k[::-1], we reverse the order of the bitstring

lAnswer = [ (k[::-1],v) for k,v in counts.items() if k != "0"*n ]

# Sort basis by probabilities
lAnswer.sort(key = lambda x: x[1], reverse=True)

Y = []
for k, v in lAnswer:
        Y.append( [ int(c) for c in k ] )

Y = Matrix(Y)

Y_transformed = Y.rref(iszerofunc=lambda x: x % 2==0)

# convert rational and negatives in rref
def mod(x,modulus):
        numer,denom = x.as_numer_denom()
        return numer*mod_inverse(denom,modulus) % modulus

# Deal with negative and fractial values
Y_new = Y_transformed[0].applyfunc(lambda x: mod(x,2))

print("The hidden period a0, a1 ... a%d only satisfies these equations:" %(n-1))
print("===============================================================\n")
rows,cols = Y_new.shape
for r in range(rows):
        Yr = [ "a"+str(i)+"" for i,v in enumerate(list(Y_new[r,:])) if v==1]
        if len(Yr) > 0:
                tStr = " + ".join(Yr)
                print(tStr, "= 0") 

# Now we need to solve this system of equations to get our period string
print("")
# a0, a2, a4
# _0_0_0
# 000000
# 010101
#        #xor
# 000000
# 010101

# Now once we have our found string, we need to double-check by XOR back to the
# y value



# Real Devices

#print("\nReal Device <device>: Resulting Values and Probabilities")
#print("===============================================\n")
#print("Simulated Runs:",shots,"\n")
