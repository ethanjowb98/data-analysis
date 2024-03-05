from matplotlib import pyplot as plot
from typing import List

def generate_parallel_boxplot(data: List[List], labels: List[str]):
    """Generates a parallel boxplot for two data lists

    Args:
        data (List[List]): must only contain two rows of list
        labels (List[str]): labels for the two rows of list
    """
    plot.figure(figsize=(10,6))
    plot.boxplot(data, labels=labels)
    plot.title("Comparison of Network Connection Stability")
    plot.xlabel("Location")
    plot.ylabel("Round-trip transit time (seconds)")
    plot.grid(True)
    plot.show()