'''
Authored by Sihyung Lee
'''

import math
import random
import logging
import os
import timeit
import statistics

MAX_ZERO_STREAK = 50

def simulate_search_with_dynamic_adjustment(N, M, M_estimated, error_margin):
    '''
    This function simulates the discovery of all solutions using Grover's algorithm, 
        while adjusting the estimate of solution count

    Input:
        N: size of search space
        M: true number of solutions
        M_estimated: an estimated of M
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

    while M_estimated > 0: # Perform discovery as long as M_estimated > 0
        #
        # Simulate one repetition of Grover's algorithm
        #
        Grover_iterations = math.floor(math.pi / 4 * math.sqrt(N / M_estimated))
        total_num_oracle_calls += (Grover_iterations + 1)  # '+1' for confirming the collected solution
        probability_measuring_solutions = math.sin((2 * Grover_iterations + 1) * math.asin(math.sqrt(M / N))) ** 2

        logging.debug(f"M_estimated = {M_estimated} with probability {probability_measuring_solutions} of measuring solutions")

        if random.random() < probability_measuring_solutions:  # random.random() returns a real number x, where 0.0 <= x < 1.0            
            #
            # Grover's algorithm discovered a solution (i.e., a success occurred)
            #
            success_n_Mestimated_history.append((1, M_estimated))
            one_streak += 1
            zero_streak = 0
            num_discovered_solutions += 1
            M_estimated = M_estimated - 1
            M = M - 1
            earlier_success_count += 1
        else:
            #
            # Grover's algorithm failed to discover a solution
            #
            success_n_Mestimated_history.append((0, M_estimated))            
            one_streak = 0        
            zero_streak += 1
            if zero_streak >= MAX_ZERO_STREAK: break  # 지정된 횟수 이상 zero streak이 반복되면 조기 종료 (보통 올바른 가설이 삭제된 경우 발생)
        
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


def simulate_search_with_baseline_method(N, M, M_estimated):
    '''
    This function simulates the discovery of all solutions using Grover's algorithm 
        without adjusting the estimate of solution count (as in previous studies)

    Input:
        N: size of search space
        M: true number of solutions
        M_estimated: an estimated of M        

    Output:
        (1) total number of solutions discovered by the baseline method
        (2) total number of oracle calls made by the baseline method    
    '''
    num_discovered_solutions = 0
    total_num_oracle_calls = 0
    zero_streak = 0

    while M_estimated > 0: # Perform discovery as long as M_estimated > 0        
        #
        # Simulate one repetition of Grover's algorithm
        #
        Grover_iterations = math.floor(math.pi / 4 * math.sqrt(N / M_estimated))
        total_num_oracle_calls += (Grover_iterations + 1)  # '+1' for confirming the collected solution
        probability_measuring_solutions = math.sin((2 * Grover_iterations + 1) * math.asin(math.sqrt(M / N))) ** 2

        logging.debug(f"M_estimated = {M_estimated} with probability {probability_measuring_solutions} of measuring solutions")

        if random.random() < probability_measuring_solutions:  # random.random() returns a real number x, where 0.0 <= x < 1.0            
            #
            # Grover's algorithm discovered a solution (i.e., a success occurred)
            #
            num_discovered_solutions += 1
            M_estimated = M_estimated - 1
            M = M - 1
            zero_streak = 0
        else:
            #
            # Grover's algorithm failed to discover a solution
            #
            zero_streak += 1            
            if zero_streak > MAX_ZERO_STREAK: break  # If no solution is collected for MAX_ZERO_STREAK consecutive attempts, terminate the repetitions, assuming that M_estimated > M

    return num_discovered_solutions, total_num_oracle_calls


if __name__ == "__main__":
    '''
    Create a log file and configure the logging level
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current Python script
    log_file_path = os.path.join(script_dir, "dynamic_adjustment_with_own_simulator.log")  # Ensure the log file is saved in the same directory as this Python script
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
    n = 8
    M = 2    
    M_estimated_list = [(0, 5), (10, 5)]

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
            num_discovered_solutions, num_oracle_calls = simulate_search_with_dynamic_adjustment(N, M, M_estimated, int(math.sqrt(N)))      
            loss_count_proposed_method.append(M - num_discovered_solutions)      
            loss_rate_proposed_method.append((M - num_discovered_solutions) / M)
            num_oracle_calls_proposed_method.append(num_oracle_calls)       
            
            #print(M, M_estimated, abs(M - M_estimated))
            num_discovered_solutions, num_oracle_calls = simulate_search_with_baseline_method(N, M, M_estimated)            
            loss_count_baseline_method.append(M - num_discovered_solutions)      
            loss_rate_baseline_method.append((M - num_discovered_solutions) / M)
            num_oracle_calls_baseline_method.append(num_oracle_calls)

    #
    # Write simulation results to the log file
    #
    info = []
    info.append(f"Experiment with custom-built simulator")    
    info.append(f"For n = {n}, N = {N}, M = {M}, and M_estimated_list = {M_estimated_list}")           
    info.append(f"\taverage # and stdev of solutions not discovered by the proposed method: {statistics.mean(loss_count_proposed_method):.2f}, {statistics.stdev(loss_count_proposed_method):.2f}")        
    info.append(f"\taverage # and stdev of oracle calls: {statistics.mean(num_oracle_calls_proposed_method):.2f}, {statistics.stdev(num_oracle_calls_proposed_method):.2f}")
    info.append(f"\taverage # and stdev of solutions not discovered by the baseline method: {statistics.mean(loss_count_baseline_method):.2f}, {statistics.stdev(loss_count_baseline_method):.2f}")        
    info.append(f"\taverage # and stdev of oracle calls: {statistics.mean(num_oracle_calls_baseline_method):.2f}, {statistics.stdev(num_oracle_calls_baseline_method):.2f}")
    info.append('')
    info_str = '\n'.join(info)
    print(info_str)
    logging.info(info_str)
