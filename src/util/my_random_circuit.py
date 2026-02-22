# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for generating random circuits."""
import numpy as np
from rustworkx import PyDiGraph

from qiskit.circuit import (
    ClassicalRegister,
    QuantumCircuit,
    CircuitInstruction,
)
from qiskit.circuit import Reset
from qiskit.circuit.classical import expr
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _BASIS_1Q, _BASIS_2Q

# (Gate class, number of qubits, number of parameters)
gates_1q_data = [
    (standard_gates.IGate, 1, 0),
    (standard_gates.SXGate, 1, 0),
    (standard_gates.XGate, 1, 0),
    (standard_gates.RZGate, 1, 1),
    (standard_gates.RGate, 1, 2),
    (standard_gates.HGate, 1, 0),
    (standard_gates.PhaseGate, 1, 1),
    (standard_gates.RXGate, 1, 1),
    (standard_gates.RYGate, 1, 1),
    (standard_gates.SGate, 1, 0),
    (standard_gates.SdgGate, 1, 0),
    (standard_gates.SXdgGate, 1, 0),
    (standard_gates.TGate, 1, 0),
    (standard_gates.TdgGate, 1, 0),
    (standard_gates.UGate, 1, 3),
    (standard_gates.U1Gate, 1, 1),
    (standard_gates.U2Gate, 1, 2),
    (standard_gates.U3Gate, 1, 3),
    (standard_gates.YGate, 1, 0),
    (standard_gates.ZGate, 1, 0),
]

gates_2q_data = [
    (standard_gates.CXGate, 2, 0),
    (standard_gates.DCXGate, 2, 0),
    (standard_gates.CHGate, 2, 0),
    (standard_gates.CPhaseGate, 2, 1),
    (standard_gates.CRXGate, 2, 1),
    (standard_gates.CRYGate, 2, 1),
    (standard_gates.CRZGate, 2, 1),
    (standard_gates.CSXGate, 2, 0),
    (standard_gates.CUGate, 2, 4),
    (standard_gates.CU1Gate, 2, 1),
    (standard_gates.CU3Gate, 2, 3),
    (standard_gates.CYGate, 2, 0),
    (standard_gates.CZGate, 2, 0),
    (standard_gates.RXXGate, 2, 1),
    (standard_gates.RYYGate, 2, 1),
    (standard_gates.RZZGate, 2, 1),
    (standard_gates.RZXGate, 2, 1),
    (standard_gates.XXMinusYYGate, 2, 2),
    (standard_gates.XXPlusYYGate, 2, 2),
    (standard_gates.ECRGate, 2, 0),
    (standard_gates.CSGate, 2, 0),
    (standard_gates.CSdgGate, 2, 0),
    (standard_gates.SwapGate, 2, 0),
    (standard_gates.iSwapGate, 2, 0),
]

def random_circuit(
    num_qubits,
    depth,
    max_operands=4,
    measure=False,
    conditional=False,
    reset=False,
    seed=None,
    num_operand_distribution: dict = None,
):
    """Generate random circuit of arbitrary size and form.

    This function will generate a random circuit by randomly selecting gates
    from the set of standard gates in :mod:`qiskit.circuit.library.standard_gates`. For example:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit.circuit.random import random_circuit

       circ = random_circuit(2, 2, measure=True)
       circ.draw(output='mpl')

    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_operands (int): maximum qubit operands of each gate (between 1 and 4)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)
        num_operand_distribution (dict): a distribution of gates that specifies the ratio
            of 1-qubit, 2-qubit, 3-qubit, ..., n-qubit gates in the random circuit. Expect a
            deviation from the specified ratios that depends on the size of the requested
            random circuit. (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
    """
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    if num_operand_distribution:
        if min(num_operand_distribution.keys()) < 1 or max(num_operand_distribution.keys()) > 4:
            raise CircuitError("'num_operand_distribution' must have keys between 1 and 4")
        for key, prob in num_operand_distribution.items():
            if key > num_qubits and prob != 0.0:
                raise CircuitError(
                    f"'num_operand_distribution' cannot have {key}-qubit gates"
                    f" for circuit with {num_qubits} qubits"
                )
        num_operand_distribution = dict(sorted(num_operand_distribution.items()))

    if not num_operand_distribution and max_operands:
        if max_operands < 1 or max_operands > 4:
            raise CircuitError("max_operands must be between 1 and 4")
        max_operands = max_operands if num_qubits > max_operands else num_qubits
        rand_dist = rng.dirichlet(
            np.ones(max_operands)
        )  # This will create a random distribution that sums to 1
        num_operand_distribution = {i + 1: rand_dist[i] for i in range(max_operands)}
        num_operand_distribution = dict(sorted(num_operand_distribution.items()))

    # Here we will use np.isclose() because very rarely there might be floating
    # point precision errors
    if not np.isclose(sum(num_operand_distribution.values()), 1):
        raise CircuitError("The sum of all the values in 'num_operand_distribution' is not 1.")

    if num_qubits == 0:
        return QuantumCircuit()

    gates_3q = [
        (standard_gates.CCXGate, 3, 0),
        (standard_gates.CSwapGate, 3, 0),
        (standard_gates.CCZGate, 3, 0),
        # (standard_gates.RCCXGate, 3, 0),
    ]

    gates_4q = [
        (standard_gates.C3SXGate, 4, 0),
        # (standard_gates.RC3XGate, 4, 0),
    ]

    gates_1q = np.array(
        gates_1q_data + [(Reset, 1, 0)] if reset else gates_1q_data,
        dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)],
    )

    gates_2q = np.array(gates_2q_data, dtype=gates_1q.dtype)
    gates_3q = np.array(gates_3q, dtype=gates_1q.dtype)
    gates_4q = np.array(gates_4q, dtype=gates_1q.dtype)

    all_gate_lists = [gates_1q, gates_2q, gates_3q, gates_4q]

    # Here we will create a list 'gates_to_consider' that will have a
    # subset of different n-qubit gates and will also create a list for
    # ratio (or probability) for each gates
    gates_to_consider = []
    distribution = []
    for n_qubits, ratio in num_operand_distribution.items():
        gate_list = all_gate_lists[n_qubits - 1]
        gates_to_consider.extend(gate_list)
        distribution.extend([ratio / len(gate_list)] * len(gate_list))

    gates = np.array(gates_to_consider, dtype=gates_1q.dtype)
    active_operand_counts = [width for width, ratio in num_operand_distribution.items() if ratio > 0]
    min_active_operands = min(active_operand_counts)

    def _sample_gate_spec_for_qubit_budget(qubit_budget):
        valid_widths = [
            width
            for width in active_operand_counts
            if width <= qubit_budget
        ]
        if not valid_widths:
            return None
        width_probs = np.array([num_operand_distribution[width] for width in valid_widths], dtype=float)
        width_probs /= width_probs.sum()
        chosen_width = int(rng.choice(valid_widths, p=width_probs))
        gate_pool = all_gate_lists[chosen_width - 1]
        return gate_pool[rng.integers(0, len(gate_pool))]

    def _build_subset_condition(bits):
        if len(bits) == 1:
            bit_val = int(rng.choice([0, 1]))
            return (bits[0], bit_val)

        condition = bits[0]
        for bit in bits[1:]:
            if rng.random() < 0.2:
                bit = expr.bit_not(bit)
            op = rng.choice(["and", "or", "xor"])
            if op == "and":
                condition = expr.bit_and(condition, bit)
            elif op == "or":
                condition = expr.bit_or(condition, bit)
            else:
                condition = expr.bit_xor(condition, bit)
        return condition

    def _append_random_conditional_ops(available_qubits, num_ops):
        for _ in range(num_ops):
            cond_gate_spec = _sample_gate_spec_for_qubit_budget(len(available_qubits))
            if cond_gate_spec is None:
                break
            cond_gate = cond_gate_spec["class"]
            cond_num_qubits = int(cond_gate_spec["num_qubits"])
            cond_num_params = int(cond_gate_spec["num_params"])
            cond_parameters = rng.uniform(0, 2 * np.pi, size=cond_num_params)
            cond_qargs = list(rng.choice(available_qubits, size=cond_num_qubits, replace=False))
            cond_operation = cond_gate(*cond_parameters)
            qc.append(CircuitInstruction(operation=cond_operation, qubits=cond_qargs))

    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    # Counter to keep track of number of different gate types
    counter = np.zeros(len(all_gate_lists) + 1, dtype=np.int64)
    total_gates = 0

    # Apply arbitrary random operations in layers across all qubits.
    for layer_number in range(depth):
        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomization routines, which can be fairly slow.

        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.

        # Due to the stochastic nature of generating a random circuit, the resulting ratios
        # may not precisely match the specified values from `num_operand_distribution`. Expect
        # greater deviations from the target ratios in quantum circuits with fewer qubits and
        # shallower depths, and smaller deviations in larger and deeper quantum circuits.
        # For more information on how the distribution changes with number of qubits and depth
        # refer to the pull request #12483 on Qiskit GitHub.

        gate_specs = rng.choice(gates, size=len(qubits), p=distribution)
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)

        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]

        slack = num_qubits - cumulative_qubits[max_index - 1]

        # Updating the counter for 1-qubit, 2-qubit, 3-qubit and 4-qubit gates
        gate_qubits = gate_specs["num_qubits"]
        counter += np.bincount(gate_qubits, minlength=len(all_gate_lists) + 1)

        total_gates += len(gate_specs)

        # Slack handling loop, this loop will add gates to fill
        # the slack while respecting the 'num_operand_distribution'
        while slack > 0:
            gate_added_flag = False

            for key, dist in sorted(num_operand_distribution.items(), reverse=True):
                if slack >= key and counter[key] / total_gates < dist:
                    gate_to_add = np.array(
                        all_gate_lists[key - 1][rng.integers(0, len(all_gate_lists[key - 1]))]
                    )
                    gate_specs = np.hstack((gate_specs, gate_to_add))
                    counter[key] += 1
                    total_gates += 1
                    slack -= key
                    gate_added_flag = True
            if not gate_added_flag:
                break

        # For efficiency in the Python loop, this uses Numpy vectorization to pre-calculate the
        # indices into the lists of qubits and parameters for every gate, and then suitably
        # randomizes those lists.
        q_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        p_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        q_indices[0] = p_indices[0] = 0
        np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
        np.cumsum(gate_specs["num_params"], out=p_indices[1:])
        parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
        rng.shuffle(qubits)

        # We've now generated everything we're going to need.  Now just to add everything.  The
        # conditional check is outside the two loops to make the more common case of no conditionals
        # faster, since in Python we don't have a compiler to do this for us.
        if conditional and layer_number != 0:
            insert_conditional_block = rng.random(size=len(gate_specs)) < 0.1
            for gate, q_start, q_end, p_start, p_end, add_conditional_block in zip(
                gate_specs["class"],
                q_indices[:-1],
                q_indices[1:],
                p_indices[:-1],
                p_indices[1:],
                insert_conditional_block,
            ):
                if add_conditional_block:
                    # Reserve enough unmeasured qubits to always place at least one operation.
                    max_num_meas = num_qubits - min_active_operands
                    if max_num_meas >= 1:
                        num_meas = int(rng.integers(1, max_num_meas + 1))
                        measured_indices = np.sort(rng.choice(num_qubits, size=num_meas, replace=False))
                        measured_set = {int(index) for index in measured_indices}
                        bits = [cr[int(index)] for index in measured_indices]
                        available_qubits = [
                            qc.qubits[index] for index in range(num_qubits) if index not in measured_set
                        ]

                        # Measure only the selected qubits and map q[i] -> c[i].
                        for index in measured_indices:
                            idx = int(index)
                            qc.measure(qc.qubits[idx], cr[idx])

                        if rng.random() < 0.5:
                            for index in measured_indices:
                                qc.reset(qc.qubits[int(index)])

                        condition = _build_subset_condition(bits)
                        num_then_ops = int(rng.integers(5, 25))
                        has_else_branch = bool(rng.random() < 0.5)
                        if has_else_branch:
                            num_else_ops = int(rng.integers(5, 25))
                            with qc.if_test(condition) as else_:
                                _append_random_conditional_ops(available_qubits, num_then_ops)
                            with else_:
                                _append_random_conditional_ops(available_qubits, num_else_ops)
                        else:
                            with qc.if_test(condition):
                                _append_random_conditional_ops(available_qubits, num_then_ops)

                operation = gate(*parameters[p_start:p_end])
                qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))
        else:
            for gate, q_start, q_end, p_start, p_end in zip(
                gate_specs["class"], q_indices[:-1], q_indices[1:], p_indices[:-1], p_indices[1:]
            ):
                operation = gate(*parameters[p_start:p_end])
                qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))
    if measure:
        qc.measure(qc.qubits, cr)

    return qc
