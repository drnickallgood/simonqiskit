import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import operator
import itertools
#from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, IBMQ
from qiskit.providers.ibmq import least_busy
from collections import OrderedDict

# AER is for simulators
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit import ClassicalRegister
from qiskit import QuantumRegister
from qiskit import execute
from qiskit import IBMQ
#from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.visualization import circuit_drawer
from sympy import Matrix, pprint, MatrixSymbol, expand, mod_inverse

from qjob import QJob

unitary_sim = Aer.get_backend('unitary_simulator')
uni_list = list()

def blackbox(simonCircuit, period_string):
        #### Blackbox Function #####
        # QP's don't care about this, we do#
        #############################
        # Copy first register to second by using CNOT gates
        for i in range(n):
                #simonCircuit.cx(qr[i],qr[n+i])
                simonCircuit.cx(qr[i],qr[n+i])
                
        # get the small index j such it's "1"
        j = -1
        #reverse the string so that it takes
        s = period_string[::-1]
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
                        simonCircuit.swap(qr[n+i],qr[n+k])    #swap gate on qubits
                        init[i], init[k] = init[k], init[i] # mark the swapped qubits
                else:
                        i += 1
                        
        # Randomly flip qubit
        # Seed random numbers for predictability / benchmark
        for i in range(n):
                if np.random.random() > 0.5:
                        simonCircuit.x(qr[n+i])

        # Added for duplicate checking

        uni_sim_result = execute(simonCircuit, unitary_sim).result()
        unitary_oracle = uni_sim_result.get_unitary(simonCircuit)
        
        # Handle empty list
        if len(uni_list) == 0:
            uni_list.append(unitary_oracle)
            print("Added to list")
            exit(1)
        else:
            # Check for duplicates
            # If duplicate oracle query, re-run oracle 
            if unitary_oracle in uni_list:
                blackbox(simonCircuit, period_string)
            else:
                # No duplicate unitary matricies, we can add to list
                uni_list.append(unitary_oracle)

        simonCircuit.barrier()
        
        return simonCircuit
        
                                        
## easily create period strings
## We want to avoid using anything with all 0's as that gives us false results
## because anything mod2 00 will give results
def create_period_str(strlen):
    str_list = list()
    for i in itertools.product([0,1],repeat=strlen):
        if "1" in ("".join(map(str,i))):
            #print("".join(map(str,i)))
            str_list.append("".join(map(str,i)))

    return str_list



#### START ####            
# hidden stringsn

period_strings_5qubit = list()
period_strings_5qubit = create_period_str(2)

period_strings_2bit = list()
period_strings_3bit = list()
period_strings_4bit = list()
period_strings_5bit = list()
period_strings_6bit = list()
period_strings_7bit = list()

# 2-bit strings
period_strings_2bit = create_period_str(2)

# 3-bit strings
period_strings_3bit = create_period_str(3)

# 4-bit strings
period_strings_4bit = create_period_str(4)

# 5-bit strings
period_strings_5bit = create_period_str(5)

# 6-bit strings
period_strings_6bit = create_period_str(6)

# 7-bit strings
period_strings_7bit = create_period_str(7)


# IBM Q stuff..
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')

circuitList = list()

## DO NOT USE ITERATION FORMULA JUST HERE FOR REF
# Iterations = # of backends tested
# iteration formula = floor(log2(num_backends * num_shots)) = 14 here
# 2-bit period strings
ranJobs = list()
backname = "local_sim"
#2bit = 12 = 36 random functions
#3bit = 54 = 372+ random functions
#4bit 
#5bit 
#6bit
#7bit
iterations = 13 
#o Jobs total = # of strings * iterations
total_jobs = iterations * len(period_strings_5bit)
job_start_idx = 1

circs = list()

def find_duplicates(circs):
    k = 0
    dup_count = 0
    while k < len(circs)-1:
        if circs[k].count_ops() == circs[k+1].count_ops():
            #print("\n=== Duplicates Found! ===")
            #print("Index:" + str(k))
            #print("Index:" + str(k+1))
            dup_count = dup_count + 1
            #print(circs[k].count_ops())
            #print(circs[k+1].count_ops())
            #print("=== End Duplcates ===")
            k = k+2
        else:
            k = k+1
            
    return dup_count
            

def generate_simon(simonCircuit, period):
    # Generate circuit
	# Assumes global simoncircuit

    # Hadamards prior to oracle
    for i in range(n):
        simonCircuit.h(qr[i])
        
    simonCircuit.barrier()
    # Oracle query
    simonCircuit = blackbox(simonCircuit, period)
    # Apply hadamards again
    for i in range(n):
        simonCircuit.h(qr[i])
        
    simonCircuit.barrier()
    # Measure qubits, maybe change to just first qubit to measure
    simonCircuit.measure(qr[0:n],cr)
    
    return simonCircuit

i = 0  
z = 0
not_done = True
np.random.seed(0)

n = len(period_strings_5bit[0])
qr = QuantumRegister(2*n, 'q')
cr = ClassicalRegister(n, 'c')
simonCircuit = QuantumCircuit(qr,cr)

while not_done:
    while i < len(period_strings_2bit):
        #print("Started main block..")
        #print(str(period_strings_2bit[i]))
        n = len(period_strings_2bit[i])
        print("Period strings: " + str(i+1) + "/" + str(len(period_strings_2bit)))
        while z < iterations:
            qr = QuantumRegister(2*n, 'q')
            cr = ClassicalRegister(n, 'c')
            simonCircuit = QuantumCircuit(qr,cr)
            # Duplicates are checked in blackbox function
            simon = generate_simon(simonCircuit, period_strings_2bit[i])
            circs.append(simon)
            z = z + 1
            print("Iterations:" + str(z) + "/" + str(iterations))
        i = i + 1
        z = 0
    not_done = False
    

