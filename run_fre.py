import subprocess

# for k in [0.1, 0.3, 0.5, 0.9, 1.0]:
#     for s in range(10):
#         command = f'python FRE.py --seed {s} --f1 {k} --data compas'
#         subprocess.run(command, shell=True)

# for s in range(10):
#     for k in [1.8, 1.9, 2, 2.1, 2.2]:
#         command = f'python FRE.py --seed {s} --f1 {k} --data german_ss'
#         subprocess.run(command, shell=True)

# for k in [0.1, 0.5, 0.7, 0.9, 1.1, 1.3]:
#     for s in range(10):
#         command = f'python FRE.py --seed {s} --f1 {k} --data adult'
#         subprocess.run(command, shell=True)
# for k in [0.01, 0.1, 0.5, 1.0, 10, 100, 1000]:
#     for s in range(10):
#         command = f'python FRE.py --seed {s} --f1 {k} --data Hospital_readmission'
#         subprocess.run(command, shell=True)

# for k in [0.01, 0.1, 1]:
#    for s in range(10):
#        command = f'python FRE.py --seed {s} --f1 {k} --data heritage'
#        subprocess.run(command, shell=True)

# for k in [1.5, 2.0]:
# # for k in [0.1, 1, 1.2, 1.5, 2.0]:
#    for s in range(10):
#        command = f'python FRE.py --seed {s} --f1 {k} --data lsac'
#        subprocess.run(command, shell=True)

# for k in [0.1, 0.5, 1.0, 10, 100, 1000]:
#     for s in range(10):
#         command = f'python FRE.py --seed {s} --f1 {k} --data bank'
#         subprocess.run(command, shell=True)

for k in [0.1, 0.5, 1.0, 1.2, 1.5]:
    for s in range(10):
        command = f'python FRE.py --seed {s} --f1 {k} --data default'
        subprocess.run(command, shell=True)