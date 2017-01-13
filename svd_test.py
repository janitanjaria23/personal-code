import numpy as np
import matplotlib.pyplot as plt
import matplotlib

la = np.linalg
words = ['I', 'like', 'enjoy', 'deep', 'learning', 'NLP', 'flying', '.']

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

eigvals = s**2 / np.cumsum(s)[-1]

plt.xlim((-2, 2))
plt.ylim((-2, 2))

for i in range(0, len(words)):
    print u[i, 0]
    print u[i, 1]
    plt.text(u[i, 0], u[i, 1], words[i])

print plt.text
plt.savefig('svd_plot.png')