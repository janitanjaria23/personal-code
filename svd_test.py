import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

eigvals = s**2 / np.cumsum(s)[-1]

fig = plt.figure(figsize=(8,5))
# sing_vals = np.arange(len(words)) + 1

# plt.plot(u, s)
# print a

for i in range(0, len(words)):
    plt.text(u[i, 0], u[i, 1], words[i])

print plt.text
plt.savefig('testplot.png')

# plt.show()
