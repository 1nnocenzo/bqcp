# Branch-Aware Quantum Constant Propagation

This repository contains a Python implementation, built on top of **Qiskit**, of a compile-time optimization pass for dynamic quantum circuits: **Branch-Aware Quantum Constant Propagation (BQCP)**.

The main implementation is in `src/ConstantPropagation.py`. The method is described in the paper:

- [Branch-Aware Quantum Constant Propagation for Dynamic Quantum Circuits](https://arxiv.org/abs/2606.02018), arXiv:2606.02018.

## What It Does

`ConstantPropagation` analyzes a Qiskit `QuantumCircuit` and produces a semantically equivalent circuit that may contain fewer operations. In particular, it can:

- remove operations that do not change the known circuit state;
- remove redundant controls from controlled gates;
- remove controlled gates that can never be activated;
- propagate known measurement and reset outcomes;
- simplify `if_else` blocks using known classical-bit information;
- track different information for different execution branches, up to a configurable limit.

The goal is to optimize dynamic circuits, i.e. circuits with mid-circuit measurements and classical control, where a non-branch-aware propagation would lose precision.

## How It Works

The analysis maintains an abstract qubit table (`UnionTable`) that maps each qubit to either:

- an explicit quantum state, when the state is still representable;
- `TOP`, when the information is unknown or too large to track.

Quantum states are represented sparsely by `QubitState`, as a map from computational-basis states to amplitudes. When a gate acts on multiple qubits, their states are combined; when a qubit becomes separable again, it is split out.

For dynamic circuits, `ConstantPropagation` also tracks:

- classical-bit states (`ZERO`, `ONE`, `NOT_KNOWN`);
- a bounded list of execution branches;
- a `max_amplitudes` threshold to avoid tracking states that become too large;
- a `max_branches` threshold to control the cost of branch-aware analysis.

When the number of branches exceeds the limit, the analysis merges them conservatively and keeps only the information common to all merged branches.

## Main Files

- `src/ConstantPropagation.py`: branch-aware optimization implementation.
- `src/util/UnionTable.py`: abstract table for qubit states.
- `src/util/QubitState.py`: sparse representation of quantum states.
- `src/util/BitState.py`: abstract state of classical bits.
- `src/util/SimplifyCondition.py`: simplification of Qiskit classical conditions.
- `run_optimization_pass.py`: script for applying the optimization to `.qpy` circuits.

## Requirements

The project requires Python and Qiskit. The main dependencies used by the implementation are:

```bash
pip install qiskit numpy
```

## How to Use It

Use `ConstantPropagation.optimize` on an existing Qiskit `QuantumCircuit`:

```python
import sys
sys.path.insert(0, "src")

from ConstantPropagation import ConstantPropagation

optimized = ConstantPropagation.optimize(
    circuit,
    max_amplitudes=512,
    max_branches=4,
)
```

Here, `circuit` is the input `QuantumCircuit`, and `optimized` is the optimized output circuit.

## Running on QPY Files

To optimize all `.qpy` circuits in a folder:

```bash
python run_optimization_pass.py <input_folder> <output_folder>
```

The script saves the optimized circuits and a JSON file with aggregate metrics. It also runs a sweep over different `max_branches` values, which is useful for comparing analysis precision and execution cost.

## Main Parameters

- `max_amplitudes`: maximum number of amplitudes tracked for an explicit quantum state. If the limit is exceeded, the state becomes `TOP`.
- `max_branches`: maximum number of branches kept separately during dynamic-circuit analysis.

Higher values can make the optimization more precise, but increase the computational cost.
