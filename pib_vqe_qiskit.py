"""
Particle in a box -> qubit Hamiltonian -> VQE, written with Qiskit (>= 1.2, also works on 2.x).

    pip install qiskit scipy numpy matplotlib
    python pib_vqe_qiskit.py                 # 3 qubits, exact (statevector) estimator
    python pib_vqe_qiskit.py --precision 0.1 # adds Gaussian sampling noise of std 0.1 to every estimate

The circuit and Pauli ordering are identical to pib_vqe_numpy.py, so the two scripts
give the same energy for the same parameters (use --crosscheck to verify).

Units: hbar = m = L = 1  ->  analytic E1 = pi^2/2.
"""
import argparse

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


def fd_hamiltonian(N, L=1.0):
    h = L / (N + 1)
    t = 1.0 / (2 * h**2)  # hbar^2/(2 m h^2) with hbar = m = 1
    return 2 * t * np.eye(N) - t * (np.eye(N, k=1) + np.eye(N, k=-1))


def build_ansatz(n, reps):
    """RY layer + CNOT chain, repeated; final RY layer (RealAmplitudes-style, linear entanglement)."""
    theta = ParameterVector("θ", n * (reps + 1))
    qc = QuantumCircuit(n)
    k = 0
    for r in range(reps + 1):
        for q in range(n):
            qc.ry(theta[k], q)
            k += 1
        if r < reps:
            for q in range(n - 1):
                qc.cx(q, q + 1)
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--precision", type=float, default=0.0,
                    help="0 = exact expectation values; >0 adds sampling-like noise")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--crosscheck", action="store_true")
    args = ap.parse_args()

    n, N = args.qubits, 2**args.qubits
    H = fd_hamiltonian(N)
    observable = SparsePauliOp.from_operator(H).simplify()   # H = sum_P c_P P
    print(f"{n} qubits, {N} grid points, {len(observable)} Pauli terms")
    print(observable)

    ansatz = build_ansatz(n, args.reps)
    print(ansatz.draw(output="text"))

    estimator = StatevectorEstimator(seed=args.seed)
    history = []

    def cost(theta):
        pub = (ansatz, observable, theta)
        if args.precision > 0:
            result = estimator.run([pub], precision=args.precision).result()
        else:
            result = estimator.run([pub]).result()
        energy = float(result[0].data.evs)
        history.append(energy)
        return energy

    rng = np.random.default_rng(args.seed)
    theta0 = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
    res = minimize(cost, theta0, method="COBYLA",
                   options={"maxiter": 4000, "rhobeg": 0.5, "tol": 1e-10})

    E_fd = np.linalg.eigvalsh(H)[0]
    print(f"\nVQE energy           : {res.fun:.8f}")
    print(f"Exact (same grid)    : {E_fd:.8f}")
    print(f"Analytic (continuum) : {np.pi**2/2:.8f}")
    print(f"Energy evaluations   : {len(history)}")

    if args.crosscheck:
        import pib_vqe_numpy as ref
        terms = ref.pauli_decompose(H)
        cache = {p: ref.pauli_matrix(p) for p in terms}
        e_np = ref.energy_exact(res.x, terms, n, args.reps, cache)
        print(f"NumPy reference energy at the same parameters: {e_np:.8f}")

    try:
        import matplotlib.pyplot as plt
        plt.plot(history, lw=1.5, label="VQE energy")
        plt.axhline(E_fd, ls="--", c="k", label="exact (grid)")
        plt.axhline(np.pi**2 / 2, ls=":", c="purple", label="analytic π²/2")
        plt.xlabel("energy evaluation")
        plt.ylabel(r"$E(\theta)$  [$\hbar^2/mL^2$]")
        plt.ylim(0, 40)
        plt.legend()
        plt.tight_layout()
        plt.savefig("vqe_convergence_qiskit.png", dpi=200)
        print("saved vqe_convergence_qiskit.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
