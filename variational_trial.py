"""
Classical variational (Rayleigh-Ritz) estimate for the particle in a box,
and a numerical check of the hard-wall virial theorem.

Trial family:  phi_alpha(x) = [x (L - x)]^alpha ,  alpha > 1/2
  - vanishes at both walls (satisfies the boundary conditions)
  - nodeless and symmetric about L/2, like the true ground state
  - alpha > 1/2 keeps the kinetic-energy integral finite

Rayleigh quotient (V = 0 inside, integrate by parts using phi(0) = phi(L) = 0):
  E(alpha) = (hbar^2/2m) * Int |phi'|^2 dx / Int |phi|^2 dx
           = (hbar^2 / m L^2) * alpha (4 alpha + 1) / (2 alpha - 1)        [closed form]
  dE/dalpha = 0  ->  8 alpha^2 - 8 alpha - 1 = 0  ->  alpha* = (2 + sqrt 6)/4
  E(alpha*) = (5/2 + sqrt 6) hbar^2/(m L^2)  ~ 4.94949  vs exact pi^2/2 ~ 4.93480

Units: hbar = m = L = 1.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


def E_closed(alpha):
    return alpha * (4 * alpha + 1) / (2 * alpha - 1)


def E_numeric(alpha, L=1.0):
    phi = lambda x: (x * (L - x)) ** alpha
    dphi = lambda x: alpha * (x * (L - x)) ** (alpha - 1) * (L - 2 * x)
    num = quad(lambda x: dphi(x) ** 2, 0, L, limit=200)[0]
    den = quad(lambda x: phi(x) ** 2, 0, L, limit=200)[0]
    return 0.5 * num / den


def main():
    E_exact = np.pi**2 / 2
    print("alpha    E_closed     E_numeric    error vs exact")
    for a in [0.75, 1.0, 1.1, (2 + np.sqrt(6)) / 4, 1.25, 1.5, 2.0]:
        print(f"{a:6.4f}  {E_closed(a):10.6f}  {E_numeric(a):10.6f}   {100*(E_closed(a)/E_exact-1):+7.3f} %")

    res = minimize_scalar(E_closed, bounds=(0.51, 5), method="bounded", options={"xatol": 1e-12})
    a_star = (2 + np.sqrt(6)) / 4
    print(f"\nnumerical minimiser alpha = {res.x:.8f}   analytic alpha* = {a_star:.8f}")
    print(f"E_min = {res.fun:.8f}   analytic 5/2 + sqrt(6) = {2.5 + np.sqrt(6):.8f}")
    print(f"exact pi^2/2 = {E_exact:.8f}  -> variational error {100*(res.fun/E_exact-1):+.4f} %  (upper bound holds)")

    # Hard-wall virial theorem for exact eigenstates:
    #   2<T>_n = (hbar^2/2m) [x |psi_n'(x)|^2]_0^L = L * F_wall,   F_wall = -dE_n/dL = 2 E_n / L
    print("\nHard-wall virial check (exact eigenfunctions, L = 1):")
    for n in range(1, 5):
        En = n**2 * np.pi**2 / 2
        psi_prime_L = np.sqrt(2) * n * np.pi * np.cos(n * np.pi)
        wall = 0.5 * 1.0 * psi_prime_L**2
        print(f"  n={n}:  2<T> = 2E_n = {2*En:9.4f}   boundary term = {wall:9.4f}   naive <x dV/dx> = 0")


if __name__ == "__main__":
    main()
