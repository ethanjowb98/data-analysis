from matplotlib import pyplot as plot
from typing import List

def generate_parallel_boxplot(data: List[List[float]], labels: List[str]):
    """Generates a parallel boxplot for two data lists

    Args:
        data (List[List[float]]): must only contain two rows of list
        labels (List[str]): labels for the two rows of list
    """
    plot.figure(figsize=(10,6))
    plot.boxplot(data, labels=labels)
    plot.scatter([1]*len(data[0]), data[0], color='black', alpha=0.7)
    plot.scatter([2]*len(data[1]), data[1], color='black', alpha=0.7)

    plot.title("Comparison of Network Connection Stability")
    plot.xlabel("Location")
    plot.ylabel("Round-trip transit time (seconds)")
    plot.grid(True)
    plot.show()