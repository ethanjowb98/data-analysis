from typing import List
import numpy

class Summary:
    
    def __init__(self) -> None:
        self.median: float
        self.minimum: float
        self.maximum: float
        self.first_quartile: float
        self.third_quartile: float

    def __str__(self):
        return f"MIN: {round(self.minimum, 2)}\nFIRST QUARTILE: {round(self.first_quartile, 2)}\nMEDIAN: {round(self.median, 2)}\nTHIRD QUARTILE: {round(self.third_quartile, 2)}\nMAX: {round(self.maximum, 2)}"

def get_five_number_summary(data: List[float]) -> Summary:
    """Prints the five number summary based on the given data

    Args:
        data (List[float])

    Return:
        Summary: contains all 5 number summary of the given data
    """
    result = Summary()
    result.median = numpy.median(data)
    result.minimum = numpy.min(data)
    result.maximum = numpy.max(data)
    
    quartiles = [0, 0.25, 0.5, 0.75, 1] 
    data_quartiles = numpy.quantile(data, quartiles)

    result.first_quartile = data_quartiles[1]
    result.third_quartile = data_quartiles[3]

    return result
