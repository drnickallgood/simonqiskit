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

def blackbox(period_string):
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

        simonCircuit.barrier()
        ### END OF BLACKBOX FUNCTION 
        
        return simonCircuit
        
def run_circuit(circuit,backend):
        # Default for this backend seems to be 1024 ibmqx2
        shots = 1024

        job = execute(simonCircuit,backend=backend, shots=shots)
        job_monitor(job,interval=2)
        results = job.result()
        return results

'''
        counts = results.get_counts()
        #print("Getting Results...\n")
        #print(qcounts)
        #print("")
        print("Submitting to IBM Q...\n")


        print("\nIBM Q Backend %s: Resulting Values and Probabilities" % qbackend)
        print("===============================================\n")
        print("Simulated Runs:",shots,"\n")

        # period, counts, prob,a0,a1,...,an
        #
        for key, val in qcounts.items():
                   prob = val / shots
                   print("Period:", key, ", Counts:", val, ", Probability:", prob)
                   
        print("")
'''
def guass_elim(results):
        # Classical post processing via Guassian elimination for the linear equations
        # Y a = 0
        # k[::-1], we reverse the order of the bitstring
        equations = list()
        lAnswer = [ (k[::-1],v) for k,v in results.get_counts().items() if k != "0"*n ]

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

        #print("\nThe hidden period a0, ... a%d only satisfies these equations:" %(n-1))
        #print("===============================================================\n")
        rows,cols = Y_new.shape
        for r in range(rows):
                        Yr = [ "a"+str(i)+"" for i,v in enumerate(list(Y_new[r,:])) if v==1]
                        if len(Yr) > 0:
                                tStr = " xor ".join(Yr)

                                #single value is 0, only xor with perid string 0 to get
                                if len(tStr) == 2:
                                        equations.append("period xor" + " 0 " + " = 0")
                                else:
                                        equations.append("period" + " xor " + tStr + " = 0")

        return equations
                                        
def print_list(results):
# Sort list by value
        sorted_x = sorted(qcounts.items(), key=operator.itemgetter(1), reverse=True)
        print("Sorted list of result strings by counts")
        print("======================================\n")
        for i in sorted_x:
                print(i)
        #print(sorted_x)
        print("")


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


## function to get dot product of result string with the period string to verify, result should be 0
#check the wikipedia for simons formula 
# DOT PRODUCT IS MOD 2 !!!!
# Result XOR ?? = 0   -- this is what we're looking for!

# We have to verify the period string with the ouptput using mod_2 addition aka XOR
# Simply xor the period string with the output string     
# Simply xor the period string with the output string, result must be 0 or 0b0
def verify_string(ostr, pstr):
    """
    Verify string with period string
    Does dot product and then mod2 addition
    """
    temp_list = list()
    # loop through outstring, make into numpy array
    for o in ostr:
        temp_list.append(int(o))
    
    ostr_arr = np.asarray(temp_list)
    temp_list.clear()
    
    # loop through period string, make into numpy array
   
    for p in pstr:
        temp_list.append(int(p))
        
    pstr_arr = np.asarray(temp_list)
    
    temp_list.clear()
    
    # Per Simosn, we do the dot product of the np arrays and then do mod 2
    results = np.dot(ostr_arr, pstr_arr)
    
    if results % 2 == 0:
        return True
     
    return False

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

# 14 qubit (broken?)
melbourne = provider.get_backend('ibmq_16_melbourne')

#5 qubit backends
ibmqx2 = provider.get_backend('ibmqx2')        # Yorktown
london = provider.get_backend('ibmq_london')
essex = provider.get_backend('ibmq_essex')
burlington = provider.get_backend('ibmq_burlington')
ourense = provider.get_backend('ibmq_ourense')
vigo = provider.get_backend('ibmq_vigo')

least = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits == 5 and not x.configuration().simulator and x.status().operational==True))

# Setup logging
# Will fail if file exists already -- because I'm lazy
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='results-2bit/' + melbourne.name() + '-2bit-36iter.txt',
                    filemode='x')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


#Nam comes back as ibmq_backend, get the part after ibmq
#least_name = least.name().split('_')[1]

#print("Least busy backend: " + least_name)

# 32 qubit qasm simulator - IBMQ
ibmq_sim = provider.get_backend('ibmq_qasm_simulator')

# Local Simulator, 
local_sim = Aer.get_backend('qasm_simulator')

circuitList = list()

backend_list = dict()
#backend_list['local_sim'] = local_sim

backend_list['ibmqx2'] = ibmqx2
backend_list['london'] = london
backend_list['essex'] = essex
backend_list['burlington'] = burlington
backend_list['ourense'] = ourense
backend_list['melbourne'] = melbourne 
backend_list['vigo'] = vigo

#backend14q_list['melbourne'] = melbourne


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
iterations = 12 
#o Jobs total = # of strings * iterations
total_jobs = iterations * len(period_strings_2bit)
job_start_idx = 1

circs = list()
dup_count = 0

# Idea here is we have are feeding hidden bitstrings and getting back results from the QC
# Create circuits
for period in period_strings_2bit:
    #print(str(period))
    n = len(period)
    # Seed random number
    #print("=== Creating Circuit ===")
    logging.info("=== Creating Circuit: " + str(period) + " ===")

    # This allows us to get consistent random functions generated for f(x)
    np.random.seed(2) ## returns 0 duplicates for 2bit stings, 36 iterations

    for k in range(iterations):
        # Generate circuit
        qr = QuantumRegister(2*n)
        cr = ClassicalRegister(n)
        simonCircuit = QuantumCircuit(qr,cr)
        
        # Hadamards prior to oracle
        for i in range(n):
            simonCircuit.h(qr[i])
            
        simonCircuit.barrier()
        # Oracle query
        simonCircuit = blackbox(period)
        # Apply hadamards again
        for i in range(n):
            simonCircuit.h(qr[i])
             
        simonCircuit.barrier()
        # Measure qubits, maybe change to just first qubit to measure
        simonCircuit.measure(qr[0:n],cr)
        circs.append(simonCircuit)
    #### end iterations loop for debugging

'''
# Check for duplicates
# We compare count_ops() to get the actual operations and order they're in
# count_ops returns OrderedDict
    
k = 0
while k < len(circs)-1:
    if circs[k].count_ops() == circs[k+1].count_ops():
        print("\n=== Duplicates Found! ===")
        print("Index:" + str(k))
        #print("Index:" + str(k+1))
        dup_count = dup_count + 1
        #print(circs[k].count_ops())
        #print(circs[k+1].count_ops())
        print("=== End Duplcates ===")
        k = k+2
    else:
        k = k+1

print("Total Circuits:" + str(len(circs)))
print("Total Duplicates:" + str(dup_count))
exit(1)
'''


# Run Circuits
logging.info("\n=== Sending data to IBMQ Backend:" + melbourne.name() + " ===\n")
for circ in circs:
    #print("Job: " + str(job_start_idx) + "/" + str(total_jobs))
    logging.info("Job: " + str(job_start_idx) + "/" + str(total_jobs))
    job = execute(circ,backend=melbourne, shots=1024)
    #job = execute(circ,backend=local_sim, shots=1024)
    job_start_idx += 1
    job_monitor(job,interval=3)
    # Store result, including period string
    qj = QJob(job,circ,melbourne.name(), period)
    ranJobs.append(qj)

# Go through and get correct vs incorrect in jobs
for qjob in ranJobs:
    results = qjob.job.result()
    counts = results.get_counts()
    equations = guass_elim(results)
    # Get period string
    pstr = qjob.getPeriod()

    obsv_strs = list()
    str_cnt = 0

    sorted_str = sorted(results.get_counts().items(), key=operator.itemgetter(1), reverse=True)

    #print("==== RAW RESULTS ====")
    #logging.info("==== RAW RESULTS ====")
    #logging.info("Period String:" + qjob.getPeriod())
    #logging.info(counts)

    # Get just the observed strings
    for string in sorted_str:
        obsv_strs.append(string[0])

    # go through and verify strings
    for o in obsv_strs:
        # Remember to re-reverse string so it's back to normal due to IBMQ Endianness
        if verify_string(o,pstr):
            # Goes through strings and counts 
            for string, count in counts.items():
                if string == o:
                    #print("===== SET CORRECT =====")
                    #print("Correct String: " + string)
                    #logging.info("Correct String: " + string)
                    #print("Correct String Counts: " + str(count))
                    qjob.setCorrect(count)        
        else:
            # lookup counts based on string
            # counts is a dict()
            for string, count in counts.items():
                if string == o:
                    # Add value to incorrect holder in object
                    #print("===== SET INCORRECT =====")
                    #print("Incorrect String: " + string)
                    #logging.info("Incorrect String: " + string)
                    #print("Incorrect String Counts: " + str(count))
                    qjob.setIncorrect(count)

total_correct = 0
total_incorrect = 0
total_runs = (1024 * iterations) * len(period_strings_2bit)

for qjob in ranJobs:
    total_correct += qjob.getCorrect()
    total_incorrect += qjob.getIncorrect() 

logging.info("\n\nTotal Runs: " + str(total_runs))
logging.info("Total Correct: " + str(total_correct))
logging.info("Prob Correct: " + str(float(total_correct) / float(total_runs)))
logging.info("Total Incorrect: " + str(total_incorrect))
logging.info("Prob Incorrect: " + str(float(total_incorrect) / float(total_runs)))

exit(1)


# Least busy backend, for individual testing
#backend_list[least_name] = least


# Make Circuits for all period strings!
#for p in period_strings_5qubit:
for p in period_strings_14qubit:

        # Circuit name = Simon_+ period string
    #circuitName = "Simon-" + p

    circuitName = p
    n = len(p)
    # For simons, we use the first n registers for control qubits
    # We use the last n registers for data qubits.. which is why we need 2*n
    qr = QuantumRegister(2*n)
    cr = ClassicalRegister(n)
    simonCircuit = QuantumCircuit(qr,cr,name=circuitName)

    # Apply hadamards prior to oracle 
    for i in range(n):
        simonCircuit.h(qr[i])
        simonCircuit.barrier()

    #call oracle for period string
    simonCircuit = blackbox(p)

    # Apply hadamards after blackbox
    for i in range(n):
        simonCircuit.h(qr[i])

    simonCircuit.barrier()

    # Measure qubits, maybe change to just first qubit to measure
    simonCircuit.measure(qr[0:n],cr)
    
    circuitList.append(simonCircuit)


# Run loop to send circuits to IBMQ..
local_sim_ranJobs = list()
ibmqx2_ranJobs = list()
london_ranJobs = list()
essex_ranJobs = list()
burlington_ranJobs = list()
ourense_ranJobs = list()
vigo_ranJobs = list()
ibmq_sim_ranJobs = list()
melbourne_ranJobs = list()


print("\n===== SENDING DATA TO IBMQ BACKENDS... =====\n")     
ranJobs = list() 

jcount = 1
jtotal = 500
for name in backend_list:
    for circuit in circuitList:
        job = execute(circuit,backend=backend_list[name], shots=1024)
        # Keep tabs on running jobs
        print("Running job on backend: " + name)
        print("Running job:  " + str(jcount) + "/" + str(jtotal))
        jcount += 1
        job_monitor(job,interval=5)
        # Custom object to hold the job, circuit, and backend
        qj = QJob(job,circuit,name)

        #print(qj.backend())
        # Append finished / ran job to list of jobs
        ranJobs.append(qj)


for qjob in ranJobs:
        # Results from each job
        results = qjob.job.result()

        # total counts from job
        counts = results.get_counts()

        # equations from each job
        equations = guass_elim(results)

        #period string encoded into name
        pstr = qjob.circuit.name

        #list of observed strings
        obs_strings = list()
        str_counts = 0
    
        # Sorted strings from each job
        sorted_str = sorted(results.get_counts().items(), key=operator.itemgetter(1), reverse=True)

        # Get just the observed strings
        for string in sorted_str:
                obs_strings.append(string[0])

        # go through and verify strings
        for o in obs_strings:
                # Remember to re-reverse string so it's back to normal due to IBMQ Endianness
                if verify_string(o,pstr):
                        for string, count in counts.items():
                            if string == o:
                                #print("===== SET CORRECT =====")
                                qjob.setCorrect(count)        
                else:
                        # lookup counts based on string
                        # counts is a dict()
                        for string, count in counts.items():
                            if string == o:
                                # Add value to incorrect holder in object
                                #print("===== SET INCORRECT =====")
                                qjob.setIncorrect(count)

        # Now we haev the stats finished, let's store them in a list based on their backend name
        if qjob.backend() == "ibmqx2":
                ibmqx2_ranJobs.append(qjob)
        elif qjob.backend() == "london":
                london_ranJobs.append(qjob)
        elif qjob.backend() == "burlington":
                burlington_ranJobs.append(qjob)
        elif qjob.backend() == "essex":
                essex_ranJobs.append(qjob)
        elif qjob.backend() == "ourense":
                ourense_ranJobs.append(qjob)
        elif qjob.backend() == "vigo":
                vigo_ranJobs.append(qjob)
        elif qjob.backend() == "ibmq_sim":
                ibmq_sim_ranJobs.append(qjob)
        elif qjob.backend() == "melbourne":
                melbourne_ranJobs.append(qjob)
        elif qjob.backend() == "local_sim":
                local_sim_ranJobs.append(qjob)
        else:
                continue



backends_5qubit_ranJobs = dict()
backends_14qubit_ranJobs = dict()
backends_sims_ranJobs = dict()

#q5b = ["ibmqx2", "vigo", "ourense", "london", "essex", "burlington"]

#q5b = ["ibmqx2"]
#q5b = ["vigo"]
#q5b = ["ourense"]
#q5b = ["london"]
q5b = ["essex"]
#q5b = ["burlington"]

q14b = ["melbourne"]

sims = ["local_sim"]
#sims = ["local_sim", "ibmq_sim"]

backends_5qubit_ranJobs['ibmqx2'] = ibmqx2_ranJobs
backends_5qubit_ranJobs['vigo'] = vigo_ranJobs
backends_5qubit_ranJobs['ourense'] = ourense_ranJobs
backends_5qubit_ranJobs['london'] = london_ranJobs
backends_5qubit_ranJobs['essex'] = essex_ranJobs
backends_5qubit_ranJobs['burlington'] = burlington_ranJobs

backends_14qubit_ranJobs['melbourne'] = melbourne_ranJobs

backends_sims_ranJobs['local_sim'] = local_sim_ranJobs
#backends_sims['ibmq_sim'] = ibmq_sim_ranJobs

# The idea here is to loop through the dictionary by using a name in the list of names above
# as such then, we can call dictionaries in a loop with that name, which contain the list of
# ran jobs
def printStats(backend, job_list):
    '''
    backend: backend name
    job_list: list of ran jobs from backend

    '''

    total_correct = 0
    total_incorrect = 0

    # Total = shots = repeitions of circuit

    # 1024 x 4 period strings we can use with 2-qubit = 4096
    # Probably make this dynamic for 14-qubit
    # 2 - 7 qubits = 142336 total runs
    total = 142336
    pcorrect = 0.00
    pincorrect = 0.00

    # Go through each job/period string's data 
    for job in job_list:
        total_correct += job.getCorrect()
        #print("Total Correct inc: " + str(total_correct))
        total_incorrect += job.getIncorrect()
        #print("Total INCorrect inc: " + str(total_incorrect))

    # This is to handle when we use a simiulator, nothing should be incorrect to avoid dividing by 0
    if total_incorrect == 0:
        pincorrect = 0.00
    else:
        pincorrect = 100*(total_incorrect / total)

    pcorrect = 100*(total_correct / total)

    print("\n===== RESULTS - " + backend + " =====\n")
    print("Total Results: " + str(total))
    print("Total Correct Results: " + str(total_correct) + " -- " + str(pcorrect) + "%") 
    print("Total Inorrect Results: " + str(total_incorrect) + " -- " + str(pincorrect) + "%")
    print("\n===================\n")


'''
for backend in sims:
    printStats(backend, backends_sims_ranJobs[backend])
'''

#printStats(least_name, backends_5qubit_ranJobs[least_name])

# for each backend name in the backend name list...

'''
for backend in q5b:
    printStats(backend, backends_5qubit_ranJobs[backend])
'''

# 14-qubit backend
for backend in q14b:
    printStats(backend, backends_14qubit_ranJobs[backend])






