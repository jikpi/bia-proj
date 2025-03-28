import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from ex3_hopfield.hopfield import HopfieldNetwork


def hopfield_solve():
    # "8"
    pattern_8 = np.array([
        0, 1, 1, 1, 0,
        0, 1, 0, 1, 0,
        0, 1, 1, 1, 0,
        0, 1, 0, 1, 0,
        0, 1, 1, 1, 0
    ])

    # "X"
    pattern_X = np.array([
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1
    ])

    # "2"
    pattern_2 = np.array([
        0, 1, 1, 1, 0,
        0, 0, 0, 1, 0,
        0, 1, 1, 1, 0,
        0, 1, 0, 0, 0,
        0, 1, 1, 1, 0
    ])

    hn = HopfieldNetwork(height=5, width=5)

    hn.learn_pattern(pattern_8)
    hn.learn_pattern(pattern_X)
    hn.learn_pattern(pattern_2)

    print("# Hopfield Network\n")
    print("Learned patterns:")

    print("\nPattern '8':")
    hn.print_pattern(pattern_8, shape=(5, 5))

    print("\nPattern 'X':")
    hn.print_pattern(pattern_X, shape=(5, 5))

    print("\nPattern '2':")
    hn.print_pattern(pattern_2, shape=(5, 5))

    # noisy "8"
    noisy_8 = np.array([
        0, 1, 0, 1, 0,
        0, 1, 0, 1, 0,
        0, 0, 0, 1, 0,
        0, 1, 0, 1, 0,
        0, 1, 1, 0, 0
    ])

    # noisy "X"
    noisy_X = np.array([
        0, 0, 0, 0, 0,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        1, 0, 0, 0, 1
    ])

    # noisy "2" (cokoliv vice noisy pada do "8")
    noisy_2 = np.array([
        0, 1, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 1, 1, 0, 0,
        0, 1, 0, 0, 0,
        0, 1, 1, 0, 0
    ])

    print("\n\nPattern Recovery:")

    print("\nNoisy '8':")
    hn.print_pattern(noisy_8, shape=(5, 5))

    recovered_8 = hn.recover_pattern(noisy_8)

    print("\nRecovered pattern:")
    hn.print_pattern(recovered_8, shape=(5, 5))

    print("\n\nNoisy 'X':")
    hn.print_pattern(noisy_X, shape=(5, 5))

    recovered_X = hn.recover_pattern(noisy_X)

    print("\nRecovered pattern:")
    hn.print_pattern(recovered_X, shape=(5, 5))

    print("\n\nNoisy '2':")
    hn.print_pattern(noisy_2, shape=(5, 5))

    recovered_2 = hn.recover_pattern(noisy_2)

    print("\nRecovered pattern:")
    hn.print_pattern(recovered_2, shape=(5, 5))

    print("\n\n# Performing hopfield performance tests")
    patterns = [pattern_8, pattern_X, pattern_2]
    graph_hopfield_performance(HopfieldNetwork, patterns)
    print("Graphs saved to Outputs folder.")


def graph_hopfield_performance(hn_class, patterns, noise_levels=10, iterations=50):
    plt.style.use('ggplot')
    sns.set_context("notebook", font_scale=1.2)

    print("Testing noise levels vs recovery success...")

    network = hn_class(height=5, width=5)
    for pattern in patterns:
        network.learn_pattern(pattern)

    noise_percents = np.linspace(0, 0.5, noise_levels)
    success_rates = []

    for noise in noise_percents:
        successes = 0

        for _ in range(iterations):
            pattern_idx = np.random.randint(0, len(patterns))
            original = patterns[pattern_idx]

            noisy = np.copy(original)
            num_flips = int(noise * len(original))
            flip_indices = np.random.choice(len(original), num_flips, replace=False)
            for idx in flip_indices:
                noisy[idx] = 1 - noisy[idx]

            recovered = network.recover_pattern(noisy)
            if np.array_equal(recovered, original):
                successes += 1

        success_rates.append(successes / iterations)

    plt.figure(figsize=(10, 6))
    plt.plot(noise_percents * 100, success_rates, 'o-', linewidth=2)
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Recovery Success Rate')
    plt.title('Pattern Recovery Success vs Noise Level')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Outputs/noise_vs_success.png', dpi=300)

    print("Testing pattern count vs recovery success...")

    all_patterns = list(patterns)
    pattern_size = len(patterns[0])

    max_patterns = 15
    while len(all_patterns) < max_patterns:
        new_pattern = np.random.choice([0, 1], size=pattern_size)
        all_patterns.append(new_pattern)

    pattern_counts = list(range(1, min(max_patterns + 1, len(all_patterns) + 1)))
    success_by_noise = {}

    test_noise_levels = [0.1, 0.2, 0.3]

    for noise in test_noise_levels:
        success_rates = []

        for num_patterns in pattern_counts:
            network = hn_class(height=5, width=5)

            for i in range(num_patterns):
                network.learn_pattern(all_patterns[i])

            successes = 0
            tests_per_pattern = max(int(iterations / num_patterns), 5)

            for i in range(num_patterns):
                original = all_patterns[i]

                for _ in range(tests_per_pattern):
                    noisy = np.copy(original)
                    num_flips = int(noise * len(original))
                    flip_indices = np.random.choice(len(original), num_flips, replace=False)
                    for idx in flip_indices:
                        noisy[idx] = 1 - noisy[idx]

                    recovered = network.recover_pattern(noisy)
                    if np.array_equal(recovered, original):
                        successes += 1

            success_rates.append(successes / (num_patterns * tests_per_pattern))

        success_by_noise[noise] = success_rates

    plt.figure(figsize=(10, 6))
    for noise, rates in success_by_noise.items():
        plt.plot(pattern_counts, rates, 'o-', linewidth=2, label=f'Noise {int(noise * 100)}%')

    plt.xlabel('Number of Patterns Stored')
    plt.ylabel('Recovery Success Rate')
    plt.title('Recovery Success vs Number of Patterns Stored')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Outputs/patterns_vs_success.png', dpi=300)
