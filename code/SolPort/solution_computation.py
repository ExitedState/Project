import itertools
import os
import pickle
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm


class SolutionComputation:
    def __init__(self, H_matrices, r = 10):
        self.H_matrices = H_matrices
        self.r = r
        self.valid_solutions = {}
        self.tasks = list(itertools.product(
            list(itertools.product([0, 1], repeat=5)),
            H_matrices.items()
        ))

    def save_results(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.valid_solutions, f)

    def load_results(self, filename):
        with open(filename, 'rb') as f:
            self.valid_solutions = pickle.load(f)
        
    def get_valid_solutions(self):
        return self.valid_solutions
    
    def compute_Valid_Solutions(self):
        results = []
        for ci_case, (combination, (covariance_matrix, H)) in tqdm(self.tasks, desc="Processing tasks", total=len(self.tasks)):
            result = self.compute_equation(
                combination,
                covariance_matrix,
                H,
                ci_case,
                self.r
            )
            results.append(result)
        
        solutions = self.filteringNone(results) # ?not only filtering None, but also make the dictionary
        self.filter_valid_solutions(solutions)
        
    @staticmethod
    def compute_equation(combination, covariance_matrix, H, ci_case, r):
        n = H.shape[0] - 2  # Number of stocks (n = 5 in this case)
        cols_to_keep = []

        # Determine which rows/columns to keep based on the current ci_case
        for i, ci in enumerate(ci_case):
            if ci == 1:  # c_i > 0, lambda_i = 0
                cols_to_keep.append(i)  # Keep c_i column
            else:  # c_i = 0, lambda_i > 0
                cols_to_keep.append(i + n)  # Keep lambda_i column
        # Add rows/columns for the additional constraints (1 and E parts)
        cols_to_keep += [H.shape[1]-2, H.shape[1] -1]  # Keep columns for 1 and expected returns (E)
        cols_to_keep.sort() #!! 2 days to solve this bug (forgot to sort) T^T
        
        # Create the reduced matrix H_reduced
        H_reduced = H[:, cols_to_keep]

        # Create the b vector
        b = np.zeros((H.shape[0], 1))
        b[-2] = 1  # Constraint on weights sum
        b[-1] = r  # Required return 'r'
        
        # Try to solve the reduced system
        try:
            x_reduced = np.linalg.solve(H_reduced, b)
            return (combination,covariance_matrix, ci_case, x_reduced, H, H_reduced, b)
        except np.linalg.LinAlgError:
            return None  # Return None if there is an error
        
    @staticmethod
    def filteringNone(results):
        print("Filtering out None results...")
        solutions = {}
        for result in results:
            if result:
                combination, covariance_matrix, ci_case, x_reduced, H, H_reduced, b = result
                if combination not in solutions:
                    solutions[combination] = []  # Initialize an empty list for this combination
                solutions[combination].append((covariance_matrix, ci_case, x_reduced, H, H_reduced, b))
        print("Successfully filtered out None results!")
        return solutions
    
    def filter_valid_solutions(self, solutions):
        print("Filtering valid solutions... (lambda>=0, c>=0)")
        progress_bar = tqdm(solutions.items(), desc="Filtering valid solutions")
        for combination, solution_list in progress_bar:
            for covariance_matrix, ci_case, x_reduced, H, H_reduced, _ in solution_list:
                checker = {}
                myport = {}
                x_values = x_reduced.flatten()  # Convert the array to a 1D array
                mu_values = {}
                mu_values['mu1'] = x_values[-2]
                mu_values['mu2'] = x_values[-1]
                increment = 0
                for i, case in enumerate(ci_case):
                    if case == 1:
                        checker[f'c{i+1}'] = x_values[increment]
                        increment += 1
                for i, case in enumerate(ci_case):
                    if case == 0:
                        checker[f'lambda{i+1}'] = x_values[increment]
                        increment += 1
                        
                for key in ['c1', 'c2', 'c3', 'c4', 'c5']:
                    if key in checker:
                        myport[key] = checker[key]
                    else:
                        myport[key] = 0
                myPort = np.array([myport['c1'], myport['c2'], myport['c3'], myport['c4'], myport['c5']]).reshape(-1,1)
                risk = np.sqrt(myPort.T @ covariance_matrix @ myPort)

                if all(value >= 0 for value in checker.values()):
                    if abs(sum(myPort) - 1) < 1e-1:
                        if combination not in self.valid_solutions:
                            self.valid_solutions[combination] = []
                        self.valid_solutions[combination].append(
                            (myPort, risk.item(), covariance_matrix, H, H_reduced, ci_case, x_reduced, checker, mu_values)
                        )
        print("Successfully filtered valid solutions!")