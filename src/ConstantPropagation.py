from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ControlledGate, Gate, Qubit, Clbit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import StatePreparation, XGate
from qiskit.circuit.classical import expr
from functools import reduce

import numpy as np

from util.UnionTable import UnionTable
from util.ActivationState import ActivationState
from util.BitState import BitState
from util.QubitState import QubitStateOrTop, EPS
from SimplifyCondition import SimplifyCondition
import random

__all__ = ["ConstantPropagation"]

def _single_qubit_matrix(instr: Instruction) -> List[complex]:
    """Return a flat list *[a, b, c, d]* representing a 2x2 unitary."""
    base = instr.base_gate if isinstance(instr, ControlledGate) else instr

    mat = Operator(base).data
    if mat.shape != (2, 2):
        raise ValueError("Instruction is not a single-qubit unitary")
    
    return [complex(mat[0, 0]), complex(mat[0, 1]), complex(mat[1, 0]), complex(mat[1, 1])]


def _two_qubit_matrix(instr: Instruction) -> List[List[complex]]:
    """Return a 4x4 nested list for two-qubit *instr*."""
    mat = Operator(instr).data

    if mat.shape != (4, 4):
        raise ValueError("Instruction is not a two-qubit unitary")
    
    return [[complex(mat[r, c]) for c in range(4)] for r in range(4)]

IGNORED_GATES: set[str] = {
    "barrier",
    "delay",
    "id"
}

UNSUPPORTED_GATES: set[str] = {
    "peres", "peresdg",
    "atru", "afalse", "multi_atru", "multi_afalse",
}

RESET_NAME = "reset"
MEASURE_NAME = "measure"
IF_ELSE_NAME = "if_else"

@dataclass
class _Branch:
    table: UnionTable
    clbit_states: dict[Clbit, BitState]

# ConstantPropagation main class
class ConstantPropagation:
    """Run the *constant-propagation* analysis/optimisation on a circuit."""

    MAX_AMPLITUDES: int = 32
    MAX_BRANCHES: int = 1

    @staticmethod
    def _clone_branch(branch: _Branch) -> _Branch:
        return _Branch(branch.table.clone(), dict(branch.clbit_states))

    @staticmethod
    def _merge_clbit_states(branches: Sequence[_Branch]) -> dict[Clbit, BitState]:
        if not branches:
            return {}
        all_clbits = set()
        for br in branches:
            all_clbits.update(br.clbit_states.keys())
        merged: dict[Clbit, BitState] = {}
        for c in all_clbits:
            states = {br.clbit_states.get(c, BitState.ZERO) for br in branches}
            merged[c] = next(iter(states)) if len(states) == 1 else BitState.NOT_KNOWN
        return merged

    @staticmethod
    def _table_signature(table: UnionTable, qubit: int) -> Optional[Tuple[Tuple[int, ...], object]]:
        reg = table[qubit]
        if reg.is_top():
            return None
        qs = reg.get_qubit_state()
        group = tuple(table.qubits_in_state(qs))
        return (group, qs)

    @classmethod
    def _merge_tables(cls, tables: Sequence[UnionTable]) -> UnionTable:
        if not tables:
            raise ValueError("No tables to merge")
        n_qubits = tables[0].size()
        merged = UnionTable(n_qubits)
        merged.qu_reg = [QubitStateOrTop() for _ in range(n_qubits)]
        visited: set[int] = set()
        base = tables[0]
        for i in range(n_qubits):
            if i in visited:
                continue
            sig = cls._table_signature(base, i)
            if sig is None:
                visited.add(i)
                continue
            group, qs = sig
            visited.update(group)
            if all(cls._table_signature(t, i) == sig for t in tables[1:]):
                qs_clone = qs.clone()
                for q in group:
                    merged.qu_reg[q] = QubitStateOrTop(qs_clone)
        return merged

    @classmethod
    def _merge_branches(cls, branches: Sequence[_Branch]) -> _Branch:
        merged_table = cls._merge_tables([br.table for br in branches])
        merged_clbits = cls._merge_clbit_states(branches)
        return _Branch(merged_table, merged_clbits)

    @staticmethod
    def _eval_tuple_condition(instr_cond, cargs, clbit_states) -> Optional[bool]:
        _, val_exp = instr_cond
        mask = 0
        expected = 0
        all_known = True
        for c in cargs:
            st = clbit_states.get(c, BitState.ZERO)
            if st == BitState.NOT_KNOWN:
                all_known = False
                continue
            mask |= (1 << c._index)
            if st == BitState.ONE:
                expected |= (1 << c._index)
        if all_known:
            return expected == val_exp
        if (val_exp & mask) != expected:
            return False
        return None

    @classmethod
    def _eval_condition(cls, instr_cond, cargs, clbit_states) -> Optional[bool]:
        if isinstance(instr_cond, tuple):
            return cls._eval_tuple_condition(instr_cond, cargs, clbit_states)
        res = SimplifyCondition.simplify(instr_cond, clbit_states)
        if res.always_true:
            return True
        if res.always_false:
            return False
        return None

    @classmethod
    def _simplify_condition_for_output(cls, instr_cond, cargs, clbit_states) -> Tuple[Optional[bool], Optional[object]]:
        if isinstance(instr_cond, tuple):
            _, val_exp = instr_cond
            mask = 0
            expected = 0
            not_determined_bits = []
            all_known = True
            for c in cargs:
                st = clbit_states.get(c, BitState.ZERO)
                if st in (BitState.ZERO, BitState.ONE):
                    mask |= (1 << c._index)
                    if st == BitState.ONE:
                        expected |= (1 << c._index)
                else:
                    all_known = False
                    not_determined_bits.append(c)
            if all_known:
                return (expected == val_exp), None
            if (val_exp & mask) != expected:
                return False, None
            if len(not_determined_bits) == 1:
                c = not_determined_bits[0]
                bit_val = 0 if (1 << c._index) & val_exp == 0 else 1
                return None, (c, bit_val)
            bits = []
            for c in not_determined_bits:
                bit_val = 0 if (1 << c._index) & val_exp == 0 else 1
                bit = c if bit_val == 1 else expr.bit_not(c)
                bits.append(bit)
            cond = reduce(expr.bit_and, bits)
            return None, cond

        res = SimplifyCondition.simplify(instr_cond, clbit_states)
        if res.always_true:
            return True, None
        if res.always_false:
            return False, None
        return None, res.expr

    @classmethod
    def _apply_ops_to_table(
        cls,
        table: UnionTable,
        ops: Sequence[Tuple[Instruction, Sequence[Qubit], Sequence[Clbit]]],
        max_amplitudes: int,
    ) -> None:
        for instr, qargs, _ in ops:
            name_lc = instr.name.lower()
            if name_lc in IGNORED_GATES:
                continue
            if name_lc in UNSUPPORTED_GATES or name_lc in (MEASURE_NAME, RESET_NAME, IF_ELSE_NAME):
                for q in qargs:
                    table.set_top(q._index)
                continue
            cls._apply_gate(table, instr, qargs, max_amplitudes)

    @classmethod
    def _handle_measurement(
        cls,
        branches: Sequence[_Branch],
        inst: Instruction,
        max_branches: int,
    ) -> Tuple[List[_Branch], bool]:
        q_ind = inst.qubits[0]._index
        c_ind = inst.clbits[0]
        new_branches: List[_Branch] = []
        keep_instr = False

        for br in branches:
            table = br.table
            clbits = br.clbit_states
            reg = table[q_ind]

            prob0 = prob1 = None
            if reg.is_qubit_state():
                qs = reg.get_qubit_state()
                idx = table.index_in_state(q_ind)
                prob0 = qs.probability_measure_zero(idx)
                prob1 = qs.probability_measure_one(idx)

            outcome = None
            if prob0 is not None and prob0 >= 1 - EPS:
                outcome = 0
            elif prob1 is not None and prob1 >= 1 - EPS:
                outcome = 1

            if outcome is not None:
                desired = BitState.ZERO if outcome == 0 else BitState.ONE
                if clbits.get(c_ind, BitState.ZERO) != desired:
                    keep_instr = True
                new_br = cls._clone_branch(br)
                new_br.table.collapse_measurement(q_ind, outcome)
                new_br.clbit_states[c_ind] = desired
                new_branches.append(new_br)
                continue

            keep_instr = True
            if len(new_branches) + 2 <= max_branches:
                for outcome in (0, 1):
                    new_br = cls._clone_branch(br)
                    if not new_br.table.collapse_measurement(q_ind, outcome):
                        continue
                    new_br.clbit_states[c_ind] = BitState.ZERO if outcome == 0 else BitState.ONE
                    new_branches.append(new_br)
            else:
                new_br = cls._clone_branch(br)
                new_br.table.set_top(q_ind)
                new_br.clbit_states[c_ind] = BitState.NOT_KNOWN
                new_branches.append(new_br)

        return new_branches, keep_instr

    @classmethod
    def _handle_reset(cls, branches: Sequence[_Branch], qubit_index: int) -> bool:
        keep_instr = False
        for br in branches:
            table = br.table
            reg = table[qubit_index]
            prob0 = None
            if reg.is_qubit_state():
                idx = table.index_in_state(qubit_index)
                prob0 = reg.get_qubit_state().probability_measure_zero(idx)
            if prob0 is None or prob0 < 1 - EPS:
                keep_instr = True
                break

        if keep_instr:
            for br in branches:
                br.table.reset_state(qubit_index)
        return keep_instr
    
    @classmethod
    def _propagate(
        cls,
        circuit: QuantumCircuit,
        max_amplitudes: int | None = None,
        table: UnionTable | None = None,
        max_branches: int | None = None,
    ) -> Tuple[UnionTable, QuantumCircuit]:
        max_amplitudes = max_amplitudes or cls.MAX_AMPLITUDES
        max_branches = cls.MAX_BRANCHES if max_branches is None else max(1, max_branches)

        # Initialize UnionTable and classical bit states
        table = table or UnionTable(circuit.num_qubits)
        clbit_states: dict[Clbit, BitState] = {c: BitState.ZERO for c in circuit.clbits}
        branches: List[_Branch] = [_Branch(table, clbit_states)]

        # Prepare new circuit
        new_circ = QuantumCircuit(*circuit.qregs, *circuit.cregs)

        # Walk through instructions
        for inst in circuit.data:
            instr = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits

            q_indices = [q._index for q in qargs]
            name_lc = instr.name.lower()

            for br in branches:
                cls._check_amplitudes(br.table, max_amplitudes)

            if all(br.table.all_top() for br in branches) and name_lc != RESET_NAME:
                new_circ.append(instr, qargs, cargs)
                continue

            # Ignored gates in the optimization
            if name_lc in IGNORED_GATES:
                new_circ.append(instr, qargs, cargs)
                continue

            # Unsupported or classically-controlled operations
            if name_lc in UNSUPPORTED_GATES:
                for br in branches:
                    for t in q_indices:
                        br.table.set_top(t)
                new_circ.append(instr, qargs, cargs)
                continue

            if name_lc == IF_ELSE_NAME:
                branches = cls._optimize_classic_controlled_operation(
                    new_circ,
                    branches,
                    inst,
                    max_amplitudes,
                    max_branches,
                )
                continue
                
            if name_lc == MEASURE_NAME: # Single measurement
                branches, keep_instr = cls._handle_measurement(branches, inst, max_branches)
                if keep_instr:
                    new_circ.append(instr, qargs, cargs)
                continue

            if name_lc == RESET_NAME:
                if cls._handle_reset(branches, q_indices[0]):
                    new_circ.append(instr, qargs, cargs)
                continue

            
            merged_table = cls._merge_tables([br.table for br in branches])
            min_contr = cls._minimize_controls(merged_table, instr, qargs)
            if min_contr is not None:
                instr_min_contr, qargs_min_contr = min_contr
                for br in branches:
                    cls._apply_gate(br.table, instr_min_contr, qargs_min_contr, max_amplitudes)
                new_circ.append(instr_min_contr, qargs_min_contr, cargs)

        merged = cls._merge_branches(branches)
        return merged.table, new_circ

    @classmethod
    def optimize(
        cls,
        circuit: QuantumCircuit,
        max_amplitudes: int | None = None,
        max_branches: int | None = None,
    ) -> QuantumCircuit:
        """Perform constant-propagation in-place on *circuit."""
        _, new_circ = cls._propagate(circuit, max_amplitudes, max_branches=max_branches)

        return new_circ
    
    @classmethod
    def _construct_branch_circuit(cls, table, instr_branch):
        if instr_branch is None:
            return []
        qc_branch = []
        for inner_inst in instr_branch:
            qc_then_instr = inner_inst.operation
            qc_then_qargs = inner_inst.qubits
            qc_then_cargs = inner_inst.clbits
        
            if table is not None:
                min_contr = cls._minimize_controls(table, qc_then_instr, qc_then_qargs)
                if min_contr is None:
                    continue # The inner operation will never be applied
                qc_then_instr, qc_then_qargs = min_contr
            
            qc_branch.append((qc_then_instr, qc_then_qargs, qc_then_cargs))
        return qc_branch
    
    @classmethod
    def _optimize_classic_controlled_operation(
        cls,
        new_circ: QuantumCircuit,
        branches: Sequence[_Branch],
        inst: Instruction,
        max_amplitudes = 1,
        max_branches = 1,
    ) -> List[_Branch]:
        instr = inst.operation
        cargs = inst.clbits
        instr_cond = instr.condition

        if not branches:
            return []

        cond_evals: List[Optional[bool]] = []
        branches_then: List[_Branch] = []
        branches_else: List[_Branch] = []
        for br in branches:
            cond_eval = cls._eval_condition(instr_cond, cargs, br.clbit_states)
            cond_evals.append(cond_eval)
            if cond_eval is True:
                branches_then.append(br)
            elif cond_eval is False:
                branches_else.append(br)
            else:
                branches_then.append(br)
                branches_else.append(br)

        table_then = cls._merge_tables([br.table for br in branches_then]) if branches_then else None
        table_else = cls._merge_tables([br.table for br in branches_else]) if branches_else else None

        qc_then = cls._construct_branch_circuit(table_then, inst.params[0] if len(inst.params) > 0 else None)
        qc_else = cls._construct_branch_circuit(table_else, inst.params[1] if len(inst.params) > 1 else None)

        merged_clbits = cls._merge_clbit_states(branches)
        cond_status, cond_expr = cls._simplify_condition_for_output(instr_cond, cargs, merged_clbits)

        if cond_status is True:
            for qc_then_instr, qc_then_qargs, qc_then_cargs in qc_then:
                new_circ.append(qc_then_instr, qc_then_qargs, qc_then_cargs)
        elif cond_status is False:
            for qc_else_instr, qc_else_qargs, qc_else_cargs in qc_else:
                new_circ.append(qc_else_instr, qc_else_qargs, qc_else_cargs)
        else:
            if len(qc_then) > 0 or len(qc_else) > 0:
                cond = cond_expr if cond_expr is not None else instr_cond
                with new_circ.if_test(cond) as else_:
                    for qc_then_instr, qc_then_qargs, qc_then_cargs in qc_then:
                        new_circ.append(qc_then_instr, qc_then_qargs, qc_then_cargs)
                if len(qc_else) > 0:
                    with else_:
                        for qc_else_instr, qc_else_qargs, qc_else_cargs in qc_else:
                            new_circ.append(qc_else_instr, qc_else_qargs, qc_else_cargs)

        next_branches: List[_Branch] = []
        for br, cond_eval in zip(branches, cond_evals):
            if cond_eval is True:
                new_br = cls._clone_branch(br)
                cls._apply_ops_to_table(new_br.table, qc_then, max_amplitudes)
                next_branches.append(new_br)
            elif cond_eval is False:
                new_br = cls._clone_branch(br)
                cls._apply_ops_to_table(new_br.table, qc_else, max_amplitudes)
                next_branches.append(new_br)
            else:
                new_br_then = cls._clone_branch(br)
                cls._apply_ops_to_table(new_br_then.table, qc_then, max_amplitudes)
                next_branches.append(new_br_then)
                new_br_else = cls._clone_branch(br)
                cls._apply_ops_to_table(new_br_else.table, qc_else, max_amplitudes)
                next_branches.append(new_br_else)

        return [cls._merge_branches(next_branches)]
    @classmethod
    def _minimize_controls(cls, table: UnionTable, instr: Instruction, qargs: Sequence[Qubit]):
        q_indices = [q._index for q in qargs]
        # Determine control and target qubits
        if isinstance(instr, ControlledGate):
            nctrl = instr.num_ctrl_qubits
            controls: List[int] = q_indices[:nctrl]
        else:
            controls = []

        # Minimize control set
        activation, min_controls = table.minimize_controls(controls)
        if activation == ActivationState.NEVER:
            return None

        instr_eff, qargs_eff = cls._rebuild_instruction(instr, qargs, min_controls)
        return (instr_eff, qargs_eff)

    @classmethod
    def _apply_gate(cls, table: UnionTable, instr: Instruction, qargs: Sequence[Qubit], max_amplitudes: int) -> None:
        q_indices = [q._index for q in qargs]
        if isinstance(instr, ControlledGate):
            nctrl = instr.num_ctrl_qubits
            controls: List[int] = q_indices[:nctrl]
            targets: List[int] = q_indices[nctrl:]
        else:
            controls = []
            targets = q_indices
            
        # Update UnionTable
        if len(targets) == 1:
            cls._apply_single_qubit_gate(table, targets[0], controls, instr)
        elif len(targets) == 2:
            cls._apply_two_qubit_gate(table, targets[0], targets[1], controls, instr)
        else:
            # Multi‑qubit gates currently unsupported
            for t in q_indices:
                table.set_top(t)

        cls._check_amplitude(table, max_amplitudes, targets[0])

    # Checks if the number of amplitudes exceeds 'max_amplitudes'
    @staticmethod
    def _check_amplitude(table: UnionTable, max_amplitudes: int, index: int) -> bool:
        reg = table[index]
        if reg.is_qubit_state() and reg.get_qubit_state().size() > max_amplitudes:
            table.set_top(index)
            return True
        return False

    @classmethod
    def _check_amplitudes(cls, table: UnionTable, max_amplitudes: int) -> None:
        for i in range(table.size()):
            cls._check_amplitude(table, max_amplitudes, i)

    @staticmethod
    def _apply_single_qubit_gate(table: UnionTable, target: int, controls: Sequence[int], instr: Instruction) -> None:
        table.combine(target, list(controls))
        if table.is_top(target):
            return
        idx_t = table.index_in_state(target)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _single_qubit_matrix(instr)
        table[target].get_qubit_state().apply_gate(idx_t, matrix, idx_ctrl)

        # Separates states of disentangled qubits after gate application
        table.separate(target)

        for c in controls:
            # Separates states of disentangled qubits after gate application
            table.separate(c)

    @staticmethod
    def _apply_two_qubit_gate(table: UnionTable, t1: int, t2: int, controls: Sequence[int], instr: Instruction) -> None:
        table.combine(t1, list(controls))
        table.combine(t1, t2)
        if table.is_top(t1):
            return
        idx1 = table.index_in_state(t1)
        idx2 = table.index_in_state(t2)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _two_qubit_matrix(instr)
        table[t1].get_qubit_state().apply_two_qubit_gate(idx1, idx2, matrix, idx_ctrl)

        # Separates states of disentangled qubits after gate application
        table.separate(t1)
        table.separate(t2)
        for c in controls:
            table.separate(c)

    # Prunes useless controls from the instruction
    @staticmethod
    def _rebuild_instruction(instr: Instruction, qargs: List[Qubit], min_controls: List[int]) -> Tuple[Instruction, List[Qubit]]:
        """Return *(instruction, qargs)* with the pruned control set."""
        if not isinstance(instr, ControlledGate):
            return instr, qargs

        original_ctrls = instr.num_ctrl_qubits
        if len(min_controls) == original_ctrls:
            return instr, qargs

        base_gate: Gate = instr.base_gate
        new_ctrl_count = len(min_controls)
        new_gate: Gate = base_gate if new_ctrl_count == 0 else base_gate.control(new_ctrl_count)

        # Reflect the order expected by Qiskit: controls first, then targets
        ctrl_qubits = [q for q in qargs[:original_ctrls] if q._index in min_controls]
        target_qubits = qargs[original_ctrls:]
        new_qargs = ctrl_qubits + list(target_qubits)

        return new_gate, new_qargs
    
    
    @staticmethod
    def _synthesize_rotation(state_vector, inverse = False) -> QuantumCircuit:
        # Ensure the input state is normalized
        state_vector = state_vector / np.linalg.norm(state_vector)
        # Get the number of qubits needed (log2 of the length of state_vector)
        n = int(np.log2(len(state_vector)))
        # Create a QuantumCircuit with n qubits
        qc = QuantumCircuit(n)
        state_preparation = StatePreparation(state_vector)
        # Append the state preparation to the quantum circuit
        qc.append(state_preparation, range(n))
        # Decompose the state preparation into individual gates
        #qc = transpile(qc, basis_gates=['h', 'cx', 'rz', 'ry'])

        if inverse:
            return qc.inverse()
        else:
            return qc
