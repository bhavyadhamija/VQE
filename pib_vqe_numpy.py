"""
Particle in a 1-D box  ->  finite-difference Hamiltonian  ->  qubit (Pauli) Hamiltonian
->  Variational Quantum Eigensolver (VQE), simulated with NumPy.

Everything here is a *classical simulation* of the quantum circuit (no quantum hardware).
It needs only numpy, scipy and (optionally) matplotlib, so it runs anywhere.

Units: hbar = m = L = 1, so energies are in units of hbar^2/(m L^2).
Analytic ground state:  E1 = pi^2/2 = 4.934802...

Qubit-ordering convention (same as Qiskit): Pauli label "P_{n-1} ... P_1 P_0",
the operator is P_{n-1} (x) ... (x) P_0, and basis index j = sum_k b_k 2^k.

Usage:
    python pib_vqe_numpy.py            # 3 qubits (8 grid points), default
    python pib_vqe_numpy.py --qubits 4 --reps 5
"""
import argparse
import itertools
import json

import numpy as np
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# 1. Physics: finite-difference Hamiltonian of the infinite square well
# --------------------------------------------------------------------------

def fd_hamiltonian(N, L=1.0, hbar=1.0, m=1.0):
    """H on N interior grid points x_j = j*h, h = L/(N+1).
    Dirichlet walls psi(0) = psi(L) = 0 are built in by dropping the end points.
    H_jk = hbar^2/(2 m h^2) * (2 delta_jk - delta_{j,k+1} - delta_{j,k-1})"""
    h = L / (N + 1)
    t = hbar**2 / (2 * m * h**2)
    H = 2 * t * np.eye(N) - t * (np.eye(N, k=1) + np.eye(N, k=-1))
    x = h * np.arange(1, N + 1)
    return H, x, h


def fd_exact_eigs(N, L=1.0, hbar=1.0, m=1.0):
    """Closed form for the discrete spectrum: E_k = (2 hbar^2/(m h^2)) sin^2(k pi / (2(N+1)))."""
    h = L / (N + 1)
    k = np.arange(1, N + 1)
    return 2 * hbar**2 / (m * h**2) * np.sin(k * np.pi / (2 * (N + 1)))**2


# --------------------------------------------------------------------------
# 2. Map the 2^n x 2^n matrix to a sum of Pauli strings
# --------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def pauli_matrix(label):
    M = np.array([[1.0 + 0j]])
    for ch in label:  # leftmost char = highest qubit (Qiskit order)
        M = np.kron(M, PAULI[ch])
    return M


def pauli_decompose(H, tol=1e-12):
    """c_P = Tr(P H) / 2^n for every Pauli string P; keep the non-zero ones."""
    n = int(np.log2(H.shape[0]))
    terms = {}
    for labels in itertools.product("IXYZ", repeat=n):
        label = "".join(labels)
        c = np.trace(pauli_matrix(label) @ H) / 2**n
        if abs(c) > tol:
            terms[label] = c.real if abs(c.imag) < tol else c
    return terms


# --------------------------------------------------------------------------
# 3. Parameterised circuit (ansatz) and a tiny statevector simulator
#    Layer = RY(theta) on every qubit, then CNOT chain 0->1->2->...; final RY layer.
#    (Same structure as Qiskit's RealAmplitudes with linear entanglement.)
# --------------------------------------------------------------------------

def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def single_qubit_op(U, q, n):
    """Embed a 1-qubit gate on qubit q (qubit 0 = rightmost factor)."""
    ops = [I2] * n
    ops[n - 1 - q] = U
    M = np.array([[1.0 + 0j]])
    for o in ops:
        M = np.kron(M, o)
    return M


def cnot_op(control, target, n):
    dim = 2**n
    M = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        bit_c = (j >> control) & 1
        k = j ^ (1 << target) if bit_c else j
        M[k, j] = 1
    return M


def n_params(n, reps):
    return n * (reps + 1)


def ansatz_state(theta, n, reps):
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0  # |00...0>
    k = 0
    for r in range(reps + 1):
        for q in range(n):
            psi = single_qubit_op(ry(theta[k]), q, n) @ psi
            k += 1
        if r < reps:
            for q in range(n - 1):
                psi = cnot_op(q, q + 1, n) @ psi
    return psi


# --------------------------------------------------------------------------
# 4. Energy estimators
# --------------------------------------------------------------------------

def energy_exact(theta, terms, n, reps, pauli_cache):
    """E(theta) = sum_P c_P <psi(theta)|P|psi(theta)>  (infinite-shot limit)."""
    psi = ansatz_state(theta, n, reps)
    return float(sum(c * np.vdot(psi, pauli_cache[p] @ psi) for p, c in terms.items()).real)


def qubitwise_commuting_groups(labels):
    """Greedy grouping: strings that agree (or have I) on every qubit can be measured together."""
    groups = []
    for lab in labels:
        for g in groups:
            if all(a == b or a == "I" or b == "I" for other in g for a, b in zip(lab, other)):
                g.append(lab)
                break
        else:
            groups.append([lab])
    return groups


Hd = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)


def energy_shots(theta, terms, n, reps, groups, shots, rng):
    """Sample bit-strings as a real device would: rotate each group into the Z basis,
    measure `shots` times, and average the +/-1 eigenvalues of every Pauli string."""
    psi = ansatz_state(theta, n, reps)
    E = terms.get("I" * n, 0.0)
    for g in groups:
        basis = ["I"] * n  # measurement basis per character position
        for lab in g:
            for i, ch in enumerate(lab):
                if ch != "I":
                    basis[i] = ch
        phi = psi.copy()
        for i, ch in enumerate(basis):
            q = n - 1 - i
            if ch == "X":
                phi = single_qubit_op(Hd, q, n) @ phi
            elif ch == "Y":
                phi = single_qubit_op(Hd @ Sdg, q, n) @ phi
        probs = np.abs(phi) ** 2
        probs /= probs.sum()
        counts = rng.multinomial(shots, probs)
        for lab in g:
            if lab == "I" * n:
                continue
            ev = 0.0
            for j, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                sign = 1
                for i, ch in enumerate(lab):
                    if ch != "I" and (j >> (n - 1 - i)) & 1:
                        sign = -sign
                ev += sign * cnt
            E += terms[lab].real * ev / shots
    return float(E)


# --------------------------------------------------------------------------
# 5. A simple SPSA optimiser (common choice for noisy, shot-based VQE)
# --------------------------------------------------------------------------

def spsa(f, x0, iters, rng, a=None, c=0.2, A=None, alpha=0.602, gamma=0.101,
         target_step=0.2, calib_samples=25):
    """Spall's SPSA. If `a` is None it is calibrated (as Qiskit's SPSA does) so that the
    first update moves each parameter by about `target_step` radians."""
    x = np.array(x0, float)
    if A is None:
        A = 0.1 * iters
    if a is None:
        g_mag = 0.0
        for _ in range(calib_samples):
            d = rng.choice([-1, 1], size=x.size)
            g_mag += abs((f(x + c * d) - f(x - c * d)) / (2 * c))
        g_mag /= calib_samples
        a = target_step * (A + 1) ** alpha / max(g_mag, 1e-12)
    hist = []
    for k in range(iters):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1, 1], size=x.size)
        g = (f(x + ck * delta) - f(x - ck * delta)) / (2 * ck) * delta
        x = x - ak * g
        hist.append(f(x))
    return x, hist


# --------------------------------------------------------------------------
# 6. Main experiment
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--starts", type=int, default=5, help="random restarts for COBYLA")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--spsa_iters", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="vqe_results.json")
    args = ap.parse_args()

    n, reps = args.qubits, args.reps
    N = 2**n
    rng = np.random.default_rng(args.seed)

    H, x, h = fd_hamiltonian(N)
    E_analytic = np.pi**2 / 2
    E_fd = np.linalg.eigvalsh(H)[0]
    assert np.isclose(E_fd, fd_exact_eigs(N)[0])

    terms = pauli_decompose(H)
    Hrec = sum(c * pauli_matrix(p) for p, c in terms.items())
    assert np.allclose(Hrec, H), "Pauli decomposition does not reproduce H"
    cache = {p: pauli_matrix(p) for p in terms}
    groups = qubitwise_commuting_groups(list(terms))

    print(f"Qubits n = {n}  ->  N = {N} interior grid points, h = {h:.5f}")
    print(f"Pauli terms: {len(terms)}  in {len(groups)} qubit-wise-commuting measurement groups")
    for p, c in sorted(terms.items()):
        print(f"   {p}: {c:+.4f}")
    print(f"Analytic continuum E1        = {E_analytic:.6f}")
    print(f"Exact diagonalisation of H_FD = {E_fd:.6f}   (discretisation error {100*(E_fd/E_analytic-1):+.3f} %)")

    # ---- Run A: noiseless statevector + COBYLA, several random starts
    P = n_params(n, reps)
    best = None
    runs = []
    for s in range(args.starts):
        th0 = rng.uniform(0, 2 * np.pi, P)
        hist = []
        f = lambda th: energy_exact(th, terms, n, reps, cache)
        res = minimize(lambda th: (hist.append(f(th)) or hist[-1]), th0,
                       method="COBYLA", options={"maxiter": 4000, "rhobeg": 0.5, "tol": 1e-10})
        runs.append({"E": float(res.fun), "nfev": int(res.nfev)})
        print(f"  start {s}: E = {res.fun:.8f}  ({res.nfev} energy evaluations)")
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x, hist)
    E_vqe, th_best, hist_best = best

    psi = ansatz_state(th_best, n, reps).real
    psi *= np.sign(psi[np.argmax(np.abs(psi))])
    w, V = np.linalg.eigh(H)
    v0 = V[:, 0] * np.sign(V[np.argmax(np.abs(V[:, 0])), 0])
    fidelity = float(abs(np.dot(psi, v0)) ** 2)

    # ---- Run B: shot-based energy at the optimum, repeated to get a spread
    E_shot_samples = [energy_shots(th_best, terms, n, reps, groups, args.shots, rng) for _ in range(30)]

    # ---- Run C: full shot-based optimisation with SPSA (noisy, like hardware without gate noise)
    th0 = rng.uniform(0, 2 * np.pi, P)
    fs = lambda th: energy_shots(th, terms, n, reps, groups, args.shots, rng)
    th_spsa, hist_spsa = spsa(fs, th0, args.spsa_iters, rng)
    E_spsa_exact = energy_exact(th_spsa, terms, n, reps, cache)
    E_spsa_shots = [energy_shots(th_spsa, terms, n, reps, groups, args.shots, rng) for _ in range(30)]

    # ---- Virial (hard-wall) check on the VQE state:  2<T> vs (hbar^2/2m) L |psi'(L)|^2
    #      continuum normalisation psi(x_j) = c_j / sqrt(h);  psi'(L) ~ (0 - psi(x_N)) / h
    T_vqe = float(psi @ H @ psi)  # V = 0 inside the box, so <T> = <H>
    slope = -psi[-1] / np.sqrt(h) / h
    wall_term = 0.5 * 1.0 * slope**2

    out = {
        "qubits": n, "grid_points": N, "reps": reps, "parameters": P,
        "pauli_terms": len(terms), "measurement_groups": len(groups),
        "pauli_coefficients": {p: float(np.real(c)) for p, c in sorted(terms.items())},
        "E_analytic": E_analytic, "E_fd_exact": float(E_fd),
        "E_vqe_statevector": float(E_vqe), "vqe_minus_fd": float(E_vqe - E_fd),
        "fidelity_with_fd_ground_state": fidelity,
        "cobyla_runs": runs, "cobyla_history_best": [float(v) for v in hist_best],
        "shots": args.shots,
        "E_shots_at_optimum_mean": float(np.mean(E_shot_samples)),
        "E_shots_at_optimum_std": float(np.std(E_shot_samples, ddof=1)),
        "spsa_iters": args.spsa_iters, "spsa_history": [float(v) for v in hist_spsa],
        "E_spsa_final_exact": float(E_spsa_exact),
        "E_spsa_final_shots_mean": float(np.mean(E_spsa_shots)),
        "E_spsa_final_shots_std": float(np.std(E_spsa_shots, ddof=1)),
        "virial_2T": 2 * T_vqe, "virial_wall_term_first_order": float(wall_term),
        "x_grid": x.tolist(), "psi_vqe": (psi / np.sqrt(h)).tolist(), "psi_fd": (v0 / np.sqrt(h)).tolist(),
    }
    print(f"VQE (statevector, best of {args.starts}) = {E_vqe:.8f}   |E_VQE - E_FD| = {abs(E_vqe-E_fd):.2e}")
    print(f"Fidelity |<psi_VQE|psi_FD>|^2 = {fidelity:.8f}")
    print(f"Shot estimate at optimum ({args.shots} shots/group, 30 repeats): "
          f"{out['E_shots_at_optimum_mean']:.4f} +/- {out['E_shots_at_optimum_std']:.4f}")
    print(f"SPSA with shots ({args.spsa_iters} iters): exact energy of final state = {E_spsa_exact:.5f}, "
          f"shot estimate = {out['E_spsa_final_shots_mean']:.4f} +/- {out['E_spsa_final_shots_std']:.4f}")
    print(f"Virial check: 2<T> = {2*T_vqe:.4f}, wall term (1st-order slope) = {wall_term:.4f}")

    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
