import matplotlib
import numpy as np
import matplotlib.pyplot as plt

x_vals = [x for x in np.arange(-2.0, 2, 0.01)]
y_square_vals = [x**2 for x in np.arange(-2.0, 2, 0.01)]
y_vals = [2*x for x in np.arange(-2.0, 2, 0.01)]

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.plot(x_vals, y_square_vals, label="$f(x) = x^2$")
plt.plot(x_vals, y_vals, label="$f'(x) = 2x$")
plt.legend()

plt.show()
