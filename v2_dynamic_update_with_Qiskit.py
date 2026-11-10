'''
Authored by Sihyung Lee
To run the codes, the following moduels must be installed:
    qiskit
    qiskit-aer
'''
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import math
import random
import logging
import os
import statistics

MAX_ZERO_STREAK = 50

def search_with_dynamic_adjustment_qiskit(N, M_estimated, answersSetOriginal, error_margin):
    '''
    This function simulates the discovery of all solutions using Grover's algorithm, 
        while adjusting the estimate of solution count

    Input:
        N: size of search space        
        M_estimated: an estimated of M
        answersSetOriginal: the set of M solutions (used to construct the oracle circuit)
        error_margin: error margin used to estimate M

    Output:
        (1) total number of solutions discovered by the proposed method
        (2) total number of oracle calls made by the proposed method    
    '''
    #
    # Initialize variables required for the simulation
    #
    num_discovered_solutions = 0    
    total_num_oracle_calls = 0

    M_original = M
    M_estimated_original = M_estimated

    #
    # Prepare variables and data structures used to maintain success/failure history
    #
    max_history_length = 20             # success_n_Mestimated_history를 기록하는 최대 길이 (모두를 기록하면 Mestimated를 adjust하는 시간이 너무 길어지므로 길이 제한)
    success_n_Mestimated_history = []   # stores a list of 2-tupes (1(success) or 0(failures), M_estimated)
    one_streak, zero_streak = 0, 0    
    earlier_success_count = 0

    #
    # Prepare variables used to trigger the adjustment of M_estimate
    #
    adjustment_interval = max_history_length    # how often do we adjust M_esitmated?
    interval = 0                                # incremented by 1 after performing Grover's algorithm and reset to 0 after adjustment of M_estimated
    adjustment_round = 0                        # total number of adjustments performed so far

    #
    # Prepare hypotheses that represent candidate M values
    #
    min_M_hypothesis = max(int(M_estimated - error_margin), 0)
    max_M_hypothesis = min(int(M_estimated + error_margin), int(N))        
    num_hypothesis = (max_M_hypothesis - min_M_hypothesis + 1)
    equal_probability_log = math.log(1 / num_hypothesis)
    M_hypotheses = [(M_hypotesis, equal_probability_log) for M_hypotesis in range(min_M_hypothesis, max_M_hypothesis + 1)]
        
    #
    # Prepare variables used to eliminate unlikely hypotheses
    #
    remove_percentage_per_round = 5
    remove_interval = 100 // remove_percentage_per_round
    
    logging.debug(f"adjustment interval = {adjustment_interval} with {len(M_hypotheses)} hypotheses")

    answersSet = answersSetOriginal.copy()     

    numDataQubits = n    
    numAncillaQubits = numDataQubits - 1
    simulator = AerSimulator()

    new_solution_just_found = True
    while M_estimated > 0: # Perform discovery as long as M_estimated > 0
        if new_solution_just_found:
            # 
            # Prepare a quantum circuit
            #
            qc = QuantumCircuit(numDataQubits + numAncillaQubits, numDataQubits) # numbers of qubits and classical bits            
            for qubit in range(numDataQubits): qc.h(qubit)  # initialization: set all data qubits into an even superposition, while ancilla qubits remain in state 0
            dataQubitIndices = [i for i in range(numDataQubits)]
            ancillaQubitIndices = [numDataQubits + i for i in range(numAncillaQubits)]
            Grover_iterations = math.floor(math.pi / 4 * math.sqrt(N / M_estimated))             
            addControlledGroverIterator(qc, None, dataQubitIndices, ancillaQubitIndices, answersSet, Grover_iterations)            
        new_solution_just_found = False

        # 
        # Run the quantum circuit and measure results
        #
        qc.measure(dataQubitIndices, range(numDataQubits))
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts(qc)        
        measuredValueInString = list(counts.keys())[0]
        measuredValueInt = int(measuredValueInString, 2)
        total_num_oracle_calls += (Grover_iterations + 1)  # '+1' for confirming the collected solution
        #print(f"{measuredValueInt} is measured")

        if measuredValueInt in answersSet:
            #
            # Grover's algorithm discovered a solution (i.e., a success occurred)
            #
            success_n_Mestimated_history.append((1, M_estimated))
            one_streak += 1
            zero_streak = 0
            num_discovered_solutions += 1
            M_estimated = M_estimated - 1            
            earlier_success_count += 1            
            del answersSet[measuredValueInt]            
            new_solution_just_found = True
            #print(f"{measuredValueInt} was found")
        else:
            #
            # Grover's algorithm failed to discover a solution
            #
            success_n_Mestimated_history.append((0, M_estimated))            
            one_streak = 0   
            zero_streak += 1            
            if zero_streak > MAX_ZERO_STREAK: break  # If no solution is collected for MAX_ZERO_STREAK consecutive attempts, terminate the repetitions, assuming that M_estimated > M

        interval += 1        
        if (interval >= adjustment_interval and one_streak < max_history_length) or M_estimated == 0:
            #
            # Begin adjusting M_estimated if ((1) and (2)) or (3)
            #      (1) interval >= adjustment_interval: after adjustment_interval repetititions of Grover's algorithms since the last adjustment
            #      (2) one_streak < max_history_length: if at least one failure occurred during the last adjustment interval
            #      (3) M_estimated == 0: perform a final adjustment to confirm whether no solution remains or not
                        
            #            
            # Use the most recent max_history_length records for adjustment to optimize performance
            #
            if len(success_n_Mestimated_history) > max_history_length: 
                success_n_Mestimated_history = success_n_Mestimated_history[len(success_n_Mestimated_history) - max_history_length:]

            omitted_successes = earlier_success_count
            for s, _ in success_n_Mestimated_history[::-1]:
                if s == 1: omitted_successes -= 1
            num_hypothesis_before = len(M_hypotheses)

            #
            # Update the probabilities of hypotheses
            #
            interval = 0            
            output_list = ['\n']
            M_hypotheses_posterior = []            
            max_probability_posterior, max_probability_hypothesis = None, None
            for hypothesis, probability_prior in M_hypotheses:
                if hypothesis - earlier_success_count < 0: continue   # A hypothesis with value < 0 cannot occur, and thus remove it

                log_likelihood = 0.0
                hypothesis_copy = hypothesis - omitted_successes
                for s, M_estimated_copy in success_n_Mestimated_history:                    
                    if s == 0:                        
                        p = math.sin((2 * math.floor(math.pi / 4 * math.sqrt(N / M_estimated_copy)) + 1) * math.asin(math.sqrt(hypothesis_copy / N))) ** 2
                        if p == 1: log_likelihood += float('-inf')
                        else: log_likelihood += math.log(1 - p)
                    else:                        
                        p = math.sin((2 * math.floor(math.pi / 4 * math.sqrt(N / M_estimated_copy)) + 1) * math.asin(math.sqrt(hypothesis_copy / N))) ** 2
                        if p == 0: log_likelihood += float('-inf')
                        else: log_likelihood += math.log(p)
                        hypothesis_copy -= 1

                probability_posterior = probability_prior + log_likelihood
                M_hypotheses_posterior.append((hypothesis_copy, probability_posterior))                

                if max_probability_posterior is None or max_probability_posterior < probability_posterior:
                    max_probability_posterior = probability_posterior
                    max_probability_hypothesis = hypothesis_copy                
            
            adjustment_round += 1
            success_n_Mestimated_history.clear()  # (v2.0) Added to make sure that once-used history is not used again

            #
            # Remove remove_percentage_per_round % of unlikely hypotheses to optimize performance            
            #
            if len(M_hypotheses_posterior) > remove_interval:   # No need to remove hypotheses if fewer than remove_interval hypotheses remain
                minimum_indices = []
                index_within_interval, minimum_within_interval, minimum_index = 0 , None, None
                for i in range(len(M_hypotheses_posterior)):
                    if minimum_within_interval == None or minimum_within_interval > M_hypotheses_posterior[i]:
                        minimum_within_interval = M_hypotheses_posterior[i]
                        minimum_index = i
                    index_within_interval += 1
                    if index_within_interval >= remove_interval:
                        minimum_indices.append(minimum_index)
                        index_within_interval, minimum_within_interval, minimum_index = 0 , None, None
                
                M_hypotheses_posterior_new = []
                from_index = 0
                for to_index in minimum_indices:
                    M_hypotheses_posterior_new.extend(M_hypotheses_posterior[from_index:to_index])
                    from_index = to_index + 1                
                M_hypotheses_posterior_new.extend(M_hypotheses_posterior[from_index:])
                M_hypotheses_posterior = M_hypotheses_posterior_new

            #
            # Normalize the posterior distribution by dividing each probability with the total likelihood,
            #   so that they sum up to 1.0
            # 
            # Also find the hypothesis with the highest likelihood            
            #
            total_likelihood = 0.0
            for _, probability_posterior in M_hypotheses_posterior:
                total_likelihood += math.e ** probability_posterior                            
            if len(M_hypotheses_posterior) == 0:
                logging.critical(f"the length of M_hyotheses_posteriror is 0")
                exit()
            if total_likelihood == 0: 
                logging.warning(f"total likelihood of posterior distribution is 0")
                logging.warning(M_hypotheses_posterior)
                logging.warning(f"M = {M}, M_estimated = {M_estimated}")
                exit()            
            total_likelihood_log = math.log(total_likelihood)
            M_hypotheses.clear()            
            for hypothesis, probability_posterior in M_hypotheses_posterior:
                final_probability_log = probability_posterior - total_likelihood_log
                M_hypotheses.append((hypothesis, final_probability_log))
                if M != hypothesis: output_list.append(f"with hypothesis={hypothesis}, the log posterior is {final_probability_log}")
                else: output_list.append(f"** with hypothesis={hypothesis}, the log posterior is {final_probability_log} **")
            num_hypothesis_after = len(M_hypotheses)

            #
            # Adjust M_estimated to the hypothesis with the highest likelihood
            #
            logging.debug(f"adjustment round {adjustment_round}")
            if max_probability_hypothesis != M_estimated:                                
                logging.debug(f"M_estimated {M_estimated} --> {max_probability_hypothesis}")                    
                #input()
                M_estimated = max_probability_hypothesis
            else:
                logging.debug(f"M_estimated {M_estimated} does not change")                    

            earlier_success_count = 0
            logging.debug(f"# of hypothesis changed from {num_hypothesis_before} to {num_hypothesis_after}")
            logging.debug('\n'.join(output_list))

    logging.debug(f"{num_discovered_solutions} out of {M_original} solutions were discovered with M_estimated = {M_estimated_original}")
    return num_discovered_solutions, total_num_oracle_calls
    

def search_with_baseline_method_qiskit(N, M_estimated, answersSetOriginal):
    '''
        This function discovers all solutions using Grover's algorithm 
            without adjusting the estimate of solution count (as in previous studies)
    
        Input:
            N: size of search space            
            M_estimated: an estimated of M
            answersSetOriginal: the set of M solutions (used to construct the oracle circuit)
    
        Output:
            (1) total number of solutions discovered by the baseline method
            (2) total number of oracle calls made by the baseline method    
    '''
    # 
    # Initialize variables
    #
    num_discovered_solutions = 0
    total_num_oracle_calls = 0
    zero_streak = 0

    answersSet = answersSetOriginal.copy()

    numDataQubits = n    
    numAncillaQubits = numDataQubits - 1
    simulator = AerSimulator()

    new_solution_just_found = True
    
    while M_estimated > 0:           
        if new_solution_just_found:
            # 
            # Prepare a quantum circuit
            #
            qc = QuantumCircuit(numDataQubits + numAncillaQubits, numDataQubits) # numbers of qubits and classical bits            
            for qubit in range(numDataQubits): qc.h(qubit)  # initialization: set all data qubits into an even superposition, while ancilla qubits remain in state 0
            dataQubitIndices = [i for i in range(numDataQubits)]
            ancillaQubitIndices = [numDataQubits + i for i in range(numAncillaQubits)]
            Grover_iterations = math.floor(math.pi / 4 * math.sqrt(N / M_estimated))             
            addControlledGroverIterator(qc, None, dataQubitIndices, ancillaQubitIndices, answersSet, Grover_iterations)            
        new_solution_just_found = False

        # 
        # Run the quantum circuit and measure results
        #
        qc.measure(dataQubitIndices, range(numDataQubits))
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts(qc)        
        measuredValueInString = list(counts.keys())[0]
        measuredValueInt = int(measuredValueInString, 2)
        total_num_oracle_calls += (Grover_iterations + 1)  # '+1' for confirming the collected solution
        #print(f"{measuredValueInt} is measured")

        if measuredValueInt in answersSet:
            #
            # Grover's algorithm discovered a solution (i.e., a success occurred)
            #
            num_discovered_solutions += 1
            M_estimated = M_estimated - 1            
            zero_streak = 0            
            del answersSet[measuredValueInt]            
            new_solution_just_found = True
            #print(f"{measuredValueInt} was found")
        else:
            #
            # Grover's algorithm failed to discover a solution
            #
            zero_streak += 1            
            if zero_streak > MAX_ZERO_STREAK: break  # If no solution is collected for MAX_ZERO_STREAK consecutive attempts, terminate the repetitions, assuming that M_estimated > M

    return num_discovered_solutions, total_num_oracle_calls


def addControlledGroverIterator(qc, controlQubitIndex, dataQubitIndices, ancillaQubitIndices, answers, numIterations):
    '''
    This function is used to create a quantum circuit that functions as a Grover's iterator
    '''
    #
    # Sanity check
    #
    assert isinstance(qc, QuantumCircuit), f"qc of class {type(qc).__name__} must be a QuantumCircuit instance"
    assert isinstance(controlQubitIndex, int) or controlQubitIndex == None, f"controlQubitIndex(={controlQubitIndex}) must be an integer or None"
    assert isinstance(dataQubitIndices, list) and all(isinstance(e, int) for e in dataQubitIndices), f"dataQubitIndices(={dataQubitIndices} must be a list with integers)"
    numDataQubits = len(dataQubitIndices)
    assert numDataQubits >= 2, f"The number of data qubits(={numDataQubits}) must be >=2 for Grover's algorithm to be effective in searching a target"
    assert isinstance(ancillaQubitIndices, list) and all(isinstance(e, int) for e in ancillaQubitIndices), f"ancillaQubitIndices(={ancillaQubitIndices} must be a list with integers)"        
    numAncillaQubits = len(ancillaQubitIndices)
    assert numAncillaQubits == numDataQubits - 1 , f"The number of ancilla qubits(={numDataQubits}) must be equal to the number of ancilla qubits(={numAncillaQubits} minus one)"
    searchSpaceSize = 2 ** numDataQubits    
    assert isinstance(numIterations, int), f"numIterations(={numIterations}) must be an integer"
        

    for _ in range(numIterations):
        #
        # Oracle - sign-invert each answer
        #
        for _, answerInBinary in answers.items():
            for bitPosition in range(numDataQubits):
                if answerInBinary[numDataQubits - bitPosition - 1] == '0': qc.x(dataQubitIndices[bitPosition])                
        
            qc.ccx(dataQubitIndices[0], dataQubitIndices[1], ancillaQubitIndices[0])
            for i in range(2, numDataQubits):
                qc.ccx(dataQubitIndices[i], ancillaQubitIndices[i-2], ancillaQubitIndices[i-1])            
            
            if controlQubitIndex == None: qc.cz(ancillaQubitIndices[-1], dataQubitIndices[-1])
            else: qc.ccz(controlQubitIndex, ancillaQubitIndices[-1], dataQubitIndices[-1])  # controlled sign-invert            

            for i in reversed(range(2, numDataQubits)):
                qc.ccx(dataQubitIndices[i], ancillaQubitIndices[i-2], ancillaQubitIndices[i-1])
            qc.ccx(dataQubitIndices[0], dataQubitIndices[1], ancillaQubitIndices[0])

            for bitPosition in range(numDataQubits):
                if answerInBinary[numDataQubits - bitPosition - 1] == '0': qc.x(dataQubitIndices[bitPosition])                

        #
        # Flip about the mean (amplitude amplification)
        #        
        for qubit in dataQubitIndices: qc.h(qubit)
        for qubit in dataQubitIndices: qc.x(qubit)

        qc.ccx(dataQubitIndices[0], dataQubitIndices[1], ancillaQubitIndices[0])
        for i in range(2, numDataQubits):
            qc.ccx(dataQubitIndices[i], ancillaQubitIndices[i-2], ancillaQubitIndices[i-1])               

        if controlQubitIndex == None: qc.cz(ancillaQubitIndices[-1], dataQubitIndices[-1])
        else: qc.ccz(controlQubitIndex, ancillaQubitIndices[-1], dataQubitIndices[-1])  # controlled sign-invert

        for i in reversed(range(2, numDataQubits)):
            qc.ccx(dataQubitIndices[i], ancillaQubitIndices[i-2], ancillaQubitIndices[i-1])
        qc.ccx(dataQubitIndices[0], dataQubitIndices[1], ancillaQubitIndices[0])

        for qubit in dataQubitIndices: qc.x(qubit)
        for qubit in dataQubitIndices: qc.h(qubit)


def generateAnswersWithEvenDistribution(searchSpaceSize, numTargets):
    '''
    Generate solutions for Scenario-even
        randomly generate solutions, so that no explicit cluter exists
    '''
    answerSet = {}
    numDataQubits = int(math.log2(searchSpaceSize))
    for _ in range(numTargets):
        r = random.randint(0, searchSpaceSize - 1)
        while r in answerSet:
            r = random.randint(0, searchSpaceSize - 1)
        answerInBinary = f"{r:0{numDataQubits}b}"
        answerSet[r] = answerInBinary

    return answerSet


if __name__ == "__main__":
    '''
    Create a log file and configure the logging level
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current Python script
    log_file_path = os.path.join(script_dir, "dynamic_adjustment_with_qiskit.log")  # Ensure the log file is saved in the same directory as this Python script
    logging.basicConfig(
        filename = log_file_path,
        level = logging.INFO,  # logging levels: debug < info < warning < error < critical        
        format = "%(asctime)s - %(levelname)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    #
    # n: number of data qubits used to represent the search space
    # M: number of solutions
    # M_estimated_list: list of (likely estimate of M by Quantum Counting, its frequency), assuming # of counting qubit m = ceiling(log_2(sqrt(N))) 
    # N: size of search space        
    #
    '''n = 6
    M = 6
    M_estimated_list = [(9, 6), (0, 3), (32, 1)]'''
    
    n = 8
    M = 2    
    M_estimated_list = [(0, 5), (10, 5)]

    '''n = 10
    M = 24
    M_estimated_list = [(9, 1), (0, 1), (39, 5), (87, 1)]'''

    '''n = 12
    M = 61
    M_estimated_list = [(39, 44), (10, 5), (0, 1), (88, 44), (156, 5), (242, 1)]'''

    '''n = 13
    M = 123
    M_estimated_list = [(78, 44), (19, 5), (0, 1), (176, 44), (311, 5), (483, 1)]'''

    '''n = 14
    M = 246
    M_estimated_list = [(157, 44), (39, 5), (0, 1), (352, 44), (623, 5), (966, 1)]'''

    N = 2 ** n

    loss_count_proposed_method = []
    loss_rate_proposed_method = []        
    num_oracle_calls_proposed_method = []        
    loss_count_baseline_method = []
    loss_rate_baseline_method = []        
    num_oracle_calls_baseline_method = []      
    for M_estimated, freq in M_estimated_list:
        for _ in range(freq):
            answersSetOriginal = generateAnswersWithEvenDistribution(N, M)
            num_discovered_solutions, num_oracle_calls = search_with_dynamic_adjustment_qiskit(N, M_estimated, answersSetOriginal, int(math.sqrt(N)))
            loss_count_proposed_method.append(M - num_discovered_solutions)      
            loss_rate_proposed_method.append((M - num_discovered_solutions) / M)
            num_oracle_calls_proposed_method.append(num_oracle_calls) 
            
            num_discovered_solutions, num_oracle_calls = search_with_baseline_method_qiskit(N, M_estimated, answersSetOriginal)
            loss_count_baseline_method.append(M - num_discovered_solutions)      
            loss_rate_baseline_method.append((M - num_discovered_solutions) / M)
            num_oracle_calls_baseline_method.append(num_oracle_calls)            

    #
    # Write simulation results to the log file
    #
    info = []    
    info.append(f"Experiment with Qiskit")
    info.append(f"For n = {n}, N = {N}, M = {M}, and M_estimated_list = {M_estimated_list}")    
    info.append(f"\taverage # and stdev of solutions not discovered by the proposed method: {statistics.mean(loss_count_proposed_method):.2f}, {statistics.stdev(loss_count_proposed_method):.2f}")        
    info.append(f"\taverage # and stdev of oracle calls: {statistics.mean(num_oracle_calls_proposed_method):.2f}, {statistics.stdev(num_oracle_calls_proposed_method):.2f}")
    info.append(f"\taverage # and stdev of solutions not discovered by the baseline method: {statistics.mean(loss_count_baseline_method):.2f}, {statistics.stdev(loss_count_baseline_method):.2f}")        
    info.append(f"\taverage # and stdev of oracle calls: {statistics.mean(num_oracle_calls_baseline_method):.2f}, {statistics.stdev(num_oracle_calls_baseline_method):.2f}")
    info.append('')
    info_str = '\n'.join(info)
    print(info_str)
    logging.info(info_str)