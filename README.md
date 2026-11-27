# Particle in a box variational method, virial theorem and VQE

Code behind the slides. Units: ħ = m = L = 1, so energies are in ħ²/(mL²) and the exact ground state is π²/2 ≈ 4.934802.

| File | What it does | Needs |
|---|---|---|
| `variational_trial.py` | Trial function [x(L−x)]^α: closed-form E(α) checked by numerical integration, optimum α* = (2+√6)/4, hard-wall virial check for n = 1–4 | numpy, scipy |
| `pib_vqe_numpy.py` | Finite-difference Hamiltonian → Pauli decomposition → RY/CNOT ansatz → VQE. Noiseless (COBYLA) and shot-based (SPSA) runs, statevector simulator written in NumPy. Produced every VQE number in the slides. | numpy, scipy |
| `pib_vqe_qiskit.py` | The same circuit and Hamiltonian in Qiskit (`SparsePauliOp`, `StatevectorEstimator`). `--crosscheck` compares with the NumPy energy. | qiskit ≥ 1.2, scipy |

```bash
python3 variational_trial.py
python3 pib_vqe_numpy.py                       # 3 qubits, 8 grid points
python3 pib_vqe_numpy.py --qubits 4 --reps 5   # 4 qubits, 16 grid points
pip install qiskit scipy matplotlib
python3 pib_vqe_qiskit.py --crosscheck
```

Results from the 3-qubit run (seed 7): exact grid energy 4.884898; VQE (noiseless) 4.884898, |ΔE| ≈ 1e-14;
VQE with 4000 shots per group and SPSA, final-state energy 4.928.

Everything here is a classical simulation of the quantum circuit. `pib_vqe_qiskit.py` was written for Qiskit 1.2+/2.x but was not executed when the slides were made, so run it yourself first.
