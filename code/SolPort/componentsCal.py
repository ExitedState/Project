from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm


class ComponentsCal:
    
    def __init__(self, combination, dividend_yield_matrix, stocks):
        self.combination = combination
        self.dividend_yield_matrix = dividend_yield_matrix
        self.stocks = stocks
    
    @staticmethod
    def calculate_covariance_matrix(selected_dividend_yield_matrix):
        """Calculate the covariance matrix for the selected stocks."""
        mean_values = np.mean(selected_dividend_yield_matrix, axis=0)
        centered_data = selected_dividend_yield_matrix - mean_values
        covariance_matrix = (centered_data.T @ centered_data) / (selected_dividend_yield_matrix.shape[0] - 1)
        return covariance_matrix, mean_values

    @staticmethod
    def calculate_bordered_hessian(covariance_matrix, mean_values):
        """Calculate the bordered Hessian matrix."""
        n = covariance_matrix.shape[0]
        expected_returns = mean_values
        negative_identity = -np.identity(n)
        combined_matrix = np.hstack((
            covariance_matrix, 
            np.where(negative_identity == -0, 0, negative_identity), 
            np.ones((n, 1)), 
            expected_returns.reshape(-1, 1)
        ))

        vertical_one = np.ones((1, n))
        vertical_zero = np.zeros((1, n+2))
        combined_vertical = np.hstack((vertical_one, vertical_zero))
        combined_matrix = np.vstack((combined_matrix, combined_vertical))

        vertical_expected = expected_returns.reshape(1, -1)
        vertical_zero = np.zeros((1, n+2))
        combined_vertical = np.hstack((vertical_expected, vertical_zero))
        combined_matrix = np.vstack((combined_matrix, combined_vertical))

        return combined_matrix

    def process_combination(self):
        selected_indices = [self.stocks.index(stock) for stock in self.combination]
        selected_dividend_yield_matrix = self.dividend_yield_matrix[:, selected_indices]

        # Step 1: Calculate the covariance matrix and mean values
        covariance_matrix, mean_values = self.calculate_covariance_matrix(selected_dividend_yield_matrix)

        # Step 2: Calculate the H matrix
        bordered_hessian = self.calculate_bordered_hessian(covariance_matrix, mean_values)

        return self.combination, covariance_matrix, bordered_hessian
    
    @staticmethod
    def run_parallel_calculations(combinations, dividend_yield_matrix, stocks, n_jobs=-1):
        # Helper function to instantiate ComponentsCal and call process_combination
        def process_combination_for_combination(combination):
            calc_instance = ComponentsCal(combination, dividend_yield_matrix, stocks)
            return calc_instance.process_combination()

        # Use joblib for parallel execution with a progress bar
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_combination_for_combination)(combination) for combination in tqdm(combinations)
        )
        
        return results