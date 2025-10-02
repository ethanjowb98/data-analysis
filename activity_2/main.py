from five_number import get_five_number_summary, Summary
from box_and_whiskers import generate_parallel_boxplot
from typing import List

location_1 = [1.6, 4.0, 3.6, 4.8, 4.2, 3.4, 5.4, 3.5, 2.8, 2.1, 3.1, 3.3, 2.2, 4.4, 4.8]
location_2 =  [3.0, 6.7, 3.9, 7.9, 4.7, 7.1, 0.5, 1.7, 6.6, 1.1, 2.0, 0.4]

def print_five_summary(data: List[float], summary_title: str):
    vertical_border = "-"*(len(summary_title)+4)
    print(vertical_border)
    print(f"| {summary_title} |")
    print(vertical_border)
    object_summary = get_five_number_summary(data)
    print(object_summary)        
    return object_summary

# def print_whisker_values(summary: Summary):
#     IQR = summary.third_quartile - summary.first_quartile
#     left_whisker = summary.first_quartile - (1.5*IQR)
#     right_whisker = summary.third_quartile + (1.5*IQR)
#     print(f"LEFT WHISKER VALUE: {left_whisker}")
#     print(f"RIGHT WHICKER VALUE: {right_whisker}")

location_1_summary = print_five_summary(location_1, "Location 1")
print()
location_2_summary = print_five_summary(location_2, "Location 2")

generate_parallel_boxplot([location_1, location_2], ["Location 1", "Location 2"])