import numpy as np


class HopfieldNetwork:
    def __init__(self, size):
        """
        Initialize a Hopfield Network

        Args:
            size (int): Number of neurons in the network
        """
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """
        Train the network using Hebbian learning rule

        Args:
            patterns (list): List of patterns to memorize, each pattern should be a numpy array
        """
        # Reset weights
        self.weights = np.zeros((self.size, self.size))

        # For each pattern
        for pattern in patterns:
            # Reshape pattern to column vector
            p = pattern.reshape(-1, 1)
            # Update weights using outer product
            self.weights += np.dot(p, p.T)

        # Remove self-connections
        np.fill_diagonal(self.weights, 0)

        # Normalize weights by number of patterns
        self.weights /= len(patterns)

    def update(self, state, num_iterations=100, update_type="async"):
        """
        Update network state

        Args:
            state (numpy.array): Initial state
            num_iterations (int): Number of iterations to run
            update_type (str): 'async' for asynchronous or 'sync' for synchronous updates

        Returns:
            numpy.array: Final state of the network
        """
        state = state.copy()

        for _ in range(num_iterations):
            if update_type == "async":
                # Update neurons in random order
                for i in np.random.permutation(self.size):
                    activation = np.dot(self.weights[i], state)
                    state[i] = 1 if activation >= 0 else -1
            else:
                # Update all neurons simultaneously
                activation = np.dot(self.weights, state)
                state = np.where(activation >= 0, 1, -1)

        return state

    def energy(self, state):
        """
        Calculate energy of the network for given state

        Args:
            state (numpy.array): Current state

        Returns:
            float: Energy of the network
        """
        return -0.5 * np.dot(np.dot(state, self.weights), state)

    def recall(self, pattern, num_iterations=100, update_type="async"):
        """
        Recall a pattern from the network

        Args:
            pattern (numpy.array): Input pattern
            num_iterations (int): Number of iterations to run
            update_type (str): 'async' for asynchronous or 'sync' for synchronous updates

        Returns:
            numpy.array: Recalled pattern
        """
        return self.update(pattern, num_iterations, update_type)


# Example usage
def create_example_patterns():
    # Create simple 4x4 patterns
    pattern1 = np.array(
        [[1, 1, 1, 1], [1, -1, -1, 1], [1, -1, -1, 1], [1, 1, 1, 1]]
    ).flatten()

    pattern2 = np.array(
        [[1, -1, -1, 1], [1, -1, -1, 1], [1, -1, -1, 1], [1, -1, -1, 1]]
    ).flatten()

    return [pattern1, pattern2]


def test_network():
    # Create and train network
    size = 4
    network = HopfieldNetwork(size)
    # patterns = create_example_patterns()
    patterns = [np.array([1, -1, -1, 1])]
    network.train(patterns)
    print(network.weights)

    # Create noisy version of pattern1
    noisy_pattern = patterns[0].copy()
    noise_positions = np.random.choice(size, size // 2, replace=False)
    noisy_pattern[noise_positions] *= -1

    # Try to recall original pattern
    recalled_pattern = network.recall(noisy_pattern)

    # Print results
    print("Original pattern:")
    print(patterns[0].reshape(2, 2))
    print("\nNoisy pattern:")
    print(noisy_pattern.reshape(2, 2))
    print("\nRecalled pattern:")
    print(recalled_pattern.reshape(2, 2))


test_network()
