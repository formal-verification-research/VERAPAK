import sys

graph = sys.argv[1]
vnnlib = sys.argv[2]

print("graph :: " + graph + "\nvnn :: " + vnnlib + "\nabstraction_strategy :: rfgsm\nverification_strategy :: discrete_search\ngranularity :: 0.00390625\ndomain_lower_bound :: -100.0\ndomain_upper_bond :: 100.0")
