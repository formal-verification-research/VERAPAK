import sys

if len(sys.argv) < 4:
    sys.exit(1)

graph = sys.argv[1]
vnnlib = sys.argv[2]
epsilon = sys.argv[3]

print(f"graph :: {graph}\nvnnlib :: {vnnlib}\nabstraction_strategy :: rfgsm\nverification_strategy :: discrete_search\ngranularity :: {epsilon}")
