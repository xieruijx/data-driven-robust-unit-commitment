# Draw a sketch map to show the idea of reconstruction
import matplotlib.pyplot as plt
import numpy as np

from utils.projection import Project

x_e, y_ep, y_en = Project().ellipse(A=2, B=-2, C=2, R=1, x0=0, y0=0)
M = np.array([[2, -1],
             [-1, 2]])
Cov = np.linalg.inv(M) * 0.25

points = np.random.multivariate_normal([0, 0], Cov, 1000)

pA0 = np.array([[1, -0.5],
               [0.7, 1],
               [-0.3, 1],
               [-1, 0],
               [-0.3, -1]]) # pA p >= pB
pB0 = np.array([-1, -2, -1, -1, -0.97])
vertices0 = Project().ineq2vertex(pA0, pB0)
x_p0 = np.append(vertices0[:, 0], vertices0[0, 0])
y_p0 = np.append(vertices0[:, 1], vertices0[0, 1])

x_w = np.array(0.546)
y_w = pA0[4, 0] * x_w - pB0[4]

pA1 = pA0
pB1 = pB0 * 0.7
vertices1 = Project().ineq2vertex(pA1, pB1)
x_p1 = np.append(vertices1[:, 0], vertices1[0, 0])
y_p1 = np.append(vertices1[:, 1], vertices1[0, 1])

# Plotting:
fig, ax = plt.subplots(1, 1)
ax.plot(points[:, 0], points[:, 1], 'k.', ms=2)
ax.plot(x_e, y_ep, 'g', label='Ellipsoid 1')
ax.plot(x_e, y_en, 'g')
ax.plot(x_p0, y_p0, label='Polyhedron 1')
ax.plot(x_w, y_w, 'gs', label='Worst case')
ax.plot(x_p1, y_p1, 'r--', label='Polyhedron 2')
ax.legend()

ax.set_aspect('equal')

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xlim([-2.3, 1.7])
ax.set_ylim([-2, 2])

# Finally, show the plot:
plt.show()