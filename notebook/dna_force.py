import matplotlib.pyplot as plt
import numpy as np



plt.style.use('my_style.mplstyle')

r = 0.59
d = 0.025
dna = 0.010
r_a = np.sqrt(2 * r * d / (1 + d/r)**2)
r_dna = np.sqrt((d+2*dna)*(2*r+2*dna))/(r*(1+(2*dna+d/2)/r))
r_2dna = np.sqrt((d+4*dna)*(2*r+4*dna))/(r*(1+(4*dna+d/2)/r))

f = 1.13e-2
epsilon = np.sqrt(np.exp(1)/2) * r_a * f 

k_cargo = 2.26e-2
v = 0.5
tau = r/v

def dna_potential_gaussian(epsilon, r, r_a, cutoff):
    if np.abs(r) <= cutoff:
        return -epsilon * np.exp(-(r**2)/(r_a**2))
    else:
        return 0

def dna_force_gaussian(epsilon, r, r_a, cutoff):
    if np.abs(r) <= cutoff:
        return -2 * epsilon * r * np.exp(-(r**2)/(r_a**2)) / (r_a**2)
    else:
        return 0

coefficient = tau / (k_cargo * r)

print("Coefficient:", epsilon*coefficient)

distances = np.linspace(0, 1.0, 1000)

fig, ax = plt.subplots()

#ax.plot(distances, [dna_force_gaussian(epsilon*coefficient, r, r_a, r_dna) for r in distances], label='gaussian potential')
ax.plot(distances, [dna_force_gaussian(epsilon*coefficient, r, r_a, r_2dna) for r in distances], label='gaussian potential')
#ax.plot(distances, [dna_force_gaussian(epsilon*coefficient, r, r_a, 1.0) for r in distances], label='gaussian potential (w/o cutoff)', ls = '--')

ax.axvline(r_a, color='black', lw=1, ls='--')
ax.axvline(r_2dna, color='black', lw=1, ls='--')

ax.text(r_a + 0.02, -1.07*epsilon*coefficient, '$\\tilde{r}_\mathrm{a}$')
ax.text(r_2dna + 0.02, -epsilon*coefficient, '$\hat{r}_\mathrm{dna}$')

ax.set_xlabel('Distance $\\tilde{r}_\mathrm{ci}$')
ax.set_ylabel('$F\\tau/R k_c$')
#ax.set_ylim(-1.1*f*coefficient, 0.1)
#ax.legend()
plt.show()

fig.savefig('figures/dna_force_gaussian.png', bbox_inches='tight')
fig.savefig('figures/dna_force_gaussian.pdf', bbox_inches='tight')


fig, ax = plt.subplots()

#ax.plot(distances, [dna_potential_gaussian(epsilon*coefficient, r, r_a, r_dna) for r in distances], label='gaussian potential')
ax.plot(distances, [dna_potential_gaussian(epsilon*coefficient, r, r_a, r_2dna) for r in distances], label='gaussian potential')
#ax.plot(distances, [dna_force_gaussian(epsilon*coefficient, r, r_a, 1.0) for r in distances], label='gaussian potential (w/o cutoff)', ls = '--')

ax.axvline(r_a, color='black', lw=1, ls='--')
ax.axvline(r_2dna, color='black', lw=1, ls='--')

ax.text(r_a + 0.02, -1.07*epsilon*coefficient, '$\\tilde{r}_\mathrm{a}$')
ax.text(r_2dna + 0.02, -epsilon*coefficient, '$\hat{r}_\mathrm{dna}$')

ax.set_xlabel('Distance $\\tilde{r}_\mathrm{ci}$')
ax.set_ylabel('$F\\tau/R k_c$')
#ax.set_ylim(-1.1*f*coefficient, 0.1)
#ax.legend()
plt.show()

fig.savefig('figures/dna_potential_gaussian.png', bbox_inches='tight')
fig.savefig('figures/dna_potential_gaussian.pdf', bbox_inches='tight')