import numpy as np
import matplotlib.pyplot as plt

la = np.linalg
words = ['I', 'like', 'deep', 'learning', 'enjoy', 'NLP', 'flying', '.']

x = np.array([[0, 2, 1, 0, 0, 0, 0, 0],
              [2, 0, 0, 1, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 1, 1, 0]])

u, s, vh = la.svd(x, full_matrices=False)

print u
print s
print vh

for i in range(0, len(words)):
    plt.text(u[i, 0], u[i, 1], words[i])
