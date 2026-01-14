from typing import List, Tuple, Dict, Optional
from copy import deepcopy

from util.QubitState import QubitState, QubitStateOrTop, EPS
from util.ActivationState import *

class UnionTable:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        # Initialize each entry to a single-qubit |0> state
        self.qu_reg: List[QubitStateOrTop] = [QubitStateOrTop(QubitState(1)) for _ in range(n_qubits)]

    def __getitem__(self, index: int) -> QubitStateOrTop:
        return self.qu_reg[index]

    def __eq__(self, other: 'UnionTable') -> bool:
        if not isinstance(other, UnionTable) or self.n_qubits != other.n_qubits:
            return False
        return all(self.qu_reg[i] == other.qu_reg[i] for i in range(self.n_qubits))

    def size(self) -> int:
        return self.n_qubits

    def all_top(self) -> bool:
        return all(reg.is_top() for reg in self.qu_reg)

    def is_top(self, index: int) -> bool:
        return self.qu_reg[index].is_top()

    def set_top(self, qubit: int) -> None:
        if self.qu_reg[qubit].is_top():
            return
        target_state = self.qu_reg[qubit].get_qubit_state()
        for i, reg in enumerate(self.qu_reg):
            if not reg.is_top() and id(reg.get_qubit_state()) == id(target_state):
                self.qu_reg[i] = QubitStateOrTop()  # TOP

    def index_in_state(self, qubit: int) -> int:
        """Return the position of `qubit` within its shared QubitState."""
        reg = self.qu_reg[qubit]
        if reg.is_top():
            return 0
        target = reg.get_qubit_state()

        idx = 0
        for i in range(qubit):
            r = self.qu_reg[i]
            if not r.is_top() and id(r.get_qubit_state()) == id(target):
                idx += 1
        return idx

    def index_in_state_list(self, qubits: List[int]) -> List[int]:
        return [self.index_in_state(q) for q in qubits]

    def qubits_in_state(self, state: QubitState) -> List[int]:
        return [
            i for i, reg in enumerate(self.qu_reg)
            if reg.is_qubit_state() and id(reg.get_qubit_state()) == id(state)
        ]

    def purity_test(self, qubit: int) -> bool:
        """Test if the qubit is in a pure (separable) state."""
        reg = self.qu_reg[qubit]
        if reg.is_top():
            return False

        qs = deepcopy(reg.get_qubit_state())
        idx = self.index_in_state(qubit)

        a0: Dict[Tuple[bool, ...], complex] = {}
        a1: Dict[Tuple[bool, ...], complex] = {}

        for key, amp in qs.state.items():
            reduced = tuple(b for i, b in enumerate(key) if i != idx)
            if key[idx]:
                a1[reduced] = amp
            else:
                a0[reduced] = amp

        if not a0 or not a1:
            return True
        if len(a0) != len(a1):
            return False

        ratio: Optional[complex] = None
        for k, v0 in a0.items():
            v1 = a1.get(k)
            if v1 is None:
                return False
            r = v0 / v1
            if ratio is None:
                ratio = r
            elif abs(r - ratio) > EPS:
                return False

        return True

    def combine(self, qubit1, qubit2_or_list) -> None:
        # Combine(int, list)
        if isinstance(qubit2_or_list, list):
            for q in qubit2_or_list:
                self.combine(qubit1, q)
            return

        # Combine two ints
        q1, q2 = qubit1, qubit2_or_list
        if q1 == q2:
            return

        r1, r2 = self.qu_reg[q1], self.qu_reg[q2]
        if r1.is_top():
            self.set_top(q2)
            return
        if r2.is_top():
            self.set_top(q1)
            return

        qs1, qs2 = r1.get_qubit_state(), r2.get_qubit_state()
        if id(qs1) == id(qs2):
            return

        idxs1 = self.qubits_in_state(qs1)
        idxs2 = self.qubits_in_state(qs2)
        new_qs = QubitState.combine(qs1, idxs1, qs2, idxs2)

        for i, reg in enumerate(self.qu_reg):
            if not reg.is_top() and (id(reg.get_qubit_state()) == id(qs1) or id(reg.get_qubit_state()) == id(qs2)):
                self.qu_reg[i] = QubitStateOrTop(new_qs)

    def is_always_one(self, q: int) -> bool:
        reg = self.qu_reg[q]
        return (not reg.is_top()) and reg.get_qubit_state().always_activated([self.index_in_state(q)])

    def is_always_zero(self, q: int) -> bool:
        reg = self.qu_reg[q]
        return (not reg.is_top()) and reg.get_qubit_state().never_activated([self.index_in_state(q)])

    def minimize_controls(self, controls: List[int]) -> Tuple[ActivationState, List[int]]:
        if not controls:
            return ActivationState.ALWAYS, []

        # If any control is always zero, never activate the gates
        for c in controls:
            if self.is_always_zero(c):
                return ActivationState.NEVER, []

        minimized: List[int] = [c for c in controls if self.qu_reg[c].is_top()]
        others = [c for c in controls if not self.qu_reg[c].is_top()]

        # Remove those always one
        others = [c for c in others if not self.is_always_one(c)]
        if not others:
            return (ActivationState.ALWAYS if not minimized else ActivationState.UNKNOWN), minimized

        # Group by shared QubitState
        groups: Dict[int, List[int]] = {}
        for c in others:
            sid = id(self.qu_reg[c].get_qubit_state())
            groups.setdefault(sid, []).append(c)

        for group in groups.values():
            if len(group) == 1:
                minimized.extend(group)
            else:
                minimized.extend(group)

        return (ActivationState.SOMETIMES if not minimized else ActivationState.UNKNOWN), minimized

    # After 'qubit' is set to 0, the other possibly entangled qubits in the state are set to top
    def reset_state(self, qubit: int) -> None:
        self.set_top(qubit)
        self.qu_reg[qubit] = QubitStateOrTop(QubitState(1))

        # Set all qubits in the same state to TOP
        self.set_top(qubit)
        # Then set the target qubit to |0>
        self.qu_reg[qubit] = QubitStateOrTop(QubitState(1))

    def collapse_measurement(self, qubit: int, outcome: int) -> bool:
        """Collapse `qubit` to outcome (0/1). Return False if outcome impossible."""
        if outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1")

        bit_val = bool(outcome)
        single = QubitState(1)
        single.clear()
        single.state[(bit_val,)] = 1 + 0j

        reg = self.qu_reg[qubit]
        if reg.is_top():
            self.qu_reg[qubit] = QubitStateOrTop(single)
            return True

        qs = reg.get_qubit_state()
        idx = self.index_in_state(qubit)
        prob = qs.probability_measure_one(idx) if bit_val else qs.probability_measure_zero(idx)
        if prob < EPS:
            return False

        if qs.get_n_qubits() == 1:
            self.qu_reg[qubit] = QubitStateOrTop(single)
            return True

        new_rest = QubitState(qs.get_n_qubits() - 1)
        new_rest.clear()
        for key, amp in qs.state.items():
            if key[idx] == bit_val:
                reduced = tuple(b for i, b in enumerate(key) if i != idx)
                new_rest.state[reduced] = new_rest.state.get(reduced, 0) + amp

        new_rest.normalize()
        new_rest.remove_zero_entries()

        group = self.qubits_in_state(qs)
        for q in group:
            if q == qubit:
                continue
            self.qu_reg[q] = QubitStateOrTop(new_rest)
        self.qu_reg[qubit] = QubitStateOrTop(single)
        return True

    def separate(self, qubit: int) -> None:
        reg = self.qu_reg[qubit]
        if reg.is_top() or not self.purity_test(qubit):
            return

        target = reg.get_qubit_state()

        # If single qubit, nothing to do
        if target.get_n_qubits() == 1:
            return

        new_rest = QubitState(target.get_n_qubits() - 1)
        new_rest.clear()

        # Collect indices of qubits that share the same QubitState object
        indices = []
        for i in range(self.n_qubits):
            if i != qubit and self.qu_reg[i].is_qubit_state() and self.qu_reg[i].get_qubit_state() is target:
                indices.append(i)

        idx = self.index_in_state(qubit)
        alpha, beta = target.amplitudes(idx)

        for key, value in target.state.items():
            # Remove the idx-th bit to form the reduced key
            reduced = tuple(b for i, b in enumerate(key) if i != idx)

            # Only set if not present OR present as exact zero (to mirror C++'s == 0 check)
            if (reduced not in new_rest.state) or (new_rest.state[reduced] == 0):
                denom = alpha if (key[idx] is False) else beta
                new_rest.state[reduced] = value / denom

        new_rest.remove_zero_entries()

        # Reassign entangled partners to the new reduced state
        for i in indices:
            self.qu_reg[i] = QubitStateOrTop(new_rest)
        
        single = QubitState(1)
        single.clear()
        single.state[(False,)] = alpha
        single.state[(True,)]  = beta
        single.remove_zero_entries()
        self.qu_reg[qubit] = QubitStateOrTop(single)

    def clone(self) -> 'UnionTable':
        new = UnionTable(self.n_qubits)
        seen: Dict[int, QubitStateOrTop] = {}
        for i, reg in enumerate(self.qu_reg):
            if reg.is_top():
                new.qu_reg[i] = QubitStateOrTop()
            else:
                qs = reg.get_qubit_state()
                key = id(qs)
                if key not in seen:
                    seen[key] = QubitStateOrTop(qs.clone())
                new.qu_reg[i] = seen[key]
        return new

    def __str__(self) -> str:
        lines = []
        for i, reg in enumerate(self.qu_reg):
            if reg.is_top():
                lines.append(f"{i}: -> TOP")
            else:
                ptr = hex(id(reg.get_qubit_state()))
                lines.append(f"{i}: -> @{ptr}: {reg.get_qubit_state()}")
        return "\n".join(lines)
