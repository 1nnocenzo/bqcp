"""
A qubit state is represented as a map bit_string -> amplitude (e.g., |00> -> 0.5, |11> -> 0.5)
"""

from copy import deepcopy
from typing import Dict, Tuple, List, Optional, Union
import numpy as np

StateKey = Tuple[bool, ...]
EPS = 1e-12 # Numerical tolerance

class QubitState:
    # State: mapping from bit-tuples to amplitude
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        # Initialise to the ground state
        init_key = tuple([False] * n_qubits)
        self.state: Dict[StateKey, complex] = {init_key: 1+0j}

    def size(self) -> int:
        return len(self.state)

    def get_n_qubits(self) -> int:
        return self.n_qubits
    
    def get_quantum_state(self) -> Dict[StateKey, complex]:
        return dict(self.state)

    def clear(self) -> None:
        self.state.clear()

    def clone(self) -> 'QubitState':
        return deepcopy(self)

    def __str__(self) -> str:
        entries = sorted(self.state.items())
        return ', '.join(f"|{''.join('1' if b else '0' for b in key)}> -> {amp}" for key, amp in entries)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: 'QubitState') -> bool:
        if not isinstance(other, QubitState):
            return False
        if self.n_qubits != other.n_qubits or self.size() != other.size():
            return False
        for k, v in self.state.items():
            if abs(v - other.state.get(k, 0)) > EPS:
                return False
        return True

    def __imul__(self, scalar: complex) -> 'QubitState':
        for k in list(self.state.keys()):
            self.state[k] *= scalar
        return self

    def __iadd__(self, other):
        for k in list(self.state):
            self.state[k] += other.state.get(k, 0)
        return self
    
    def __iter__(self):
        return iter(self.state.items())

    def to_state_vector(self) -> List[complex]:
        size = 1 << self.n_qubits
        vec = [0j] * size
        for key, amp in self.state.items():
            idx = sum((1 << i) if bit else 0 for i, bit in enumerate(key))
            vec[idx] = amp
        return vec

    @staticmethod
    def from_vector(vector: List[complex], n_qubits: int) -> 'QubitState':
        qs = QubitState(n_qubits)
        qs.clear()
        for idx, amp in enumerate(vector):
            if abs(amp) > EPS:
                # convert idx to bit tuple
                key = tuple(bool((idx >> i) & 1) for i in range(n_qubits))
                qs.state[key] = amp
        return qs
    
    def norm(self) -> float:
        return sum(abs(v)**2 for v in self.state.values())

    def normalize(self) -> None:
        nor = np.sqrt(self.norm())
        if nor > 0:
            self *= (1 / nor)

    def remove_zero_entries(self) -> None:
        zeros = [k for k, v in self.state.items() if abs(v) < EPS]
        total_removed = sum(abs(self.state[k])**2 for k in zeros)
        for k in zeros:
            del self.state[k]
        if total_removed < 1:
            factor = 1/ np.sqrt(1 - total_removed)
            self *= factor

    def probability_measure_zero(self, index: int) -> float:
        return self._probability_measure_x(index, False)

    def probability_measure_one(self, index: int) -> float:
        return self._probability_measure_x(index, True)

    def _probability_measure_x(self, index: int, x: bool) -> float:
        if all(k[index] == x for k in self.state):
            return 1.0
        return sum(abs(v)**2 for k, v in self.state.items() if k[index] == x)

    def amplitudes(self, index: int) -> Tuple[complex, complex]:
        if self.n_qubits == 1:
            alpha = self.state.get((False,), 0)
            beta  = self.state.get((True,), 0)
            return alpha, beta
        
        # Manage multi-qubit states
        dm = np.zeros((2,2), dtype=complex)
        
        for (k1, v1) in self.state.items():
            for (k2, v2) in self.state.items():

                k1_bit = 1 if k1[index] else 0
                k2_bit = 1 if k2[index] else 0

                # Remove that bit and compare the remaining substrings
                k1_rest = k1[:index] + k1[index+1:]
                k2_rest = k2[:index] + k2[index+1:]

                if k1_rest == k2_rest:
                    dm[k1_bit, k2_bit] += complex(v1) * np.conj(complex(v2))

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(dm)
        # Pick eigenvector with eigenvalue 1
        idx = int(np.argmin(np.abs(eigvals - 1)))
        vec = eigvecs[:, idx] # Contains the amplitudes alpha and beta

        return vec[0], vec[1]

    def apply_gate(self, target: int, matrix: List[complex], controls: Optional[List[int]] = None) -> None:
        if controls:
            activated = QubitState(self.n_qubits)
            activated.clear()
            deactivated = QubitState(self.n_qubits)
            deactivated.clear()
            for k, v in self.state.items():
                if all(k[c] for c in controls):
                    activated.state[k] = v
                else:
                    deactivated.state[k] = v
            activated.apply_gate(target, matrix)
            self.state = activated.state
            for k, v in deactivated.state.items():
                self.state[k] = self.state.get(k, 0) + v
            self.remove_zero_entries()
            return
        
        new_state: Dict[StateKey, complex] = {}
        for k, v in self.state.items():
            if abs(v) < EPS:
                continue
            # Matrix: [a, b, c, d] short for [[a, b], [c, d]]
            a, b, c, d = matrix[0], matrix[1], matrix[2], matrix[3]
            if k[target]:
                # Case |1>
                if abs(b) > EPS:
                    nk = list(k)
                    nk[target] = False
                    nk = tuple(nk)
                    new_state[nk] = new_state.get(nk, 0) + v * b
                if abs(d) > EPS:
                    new_state[k] = new_state.get(k, 0) + v * d
            else:
                # Case |0>
                if abs(a) > 0:
                    new_state[k] = new_state.get(k, 0) + v * a
                if abs(c) > 0:
                    nk = list(k)
                    nk[target] = True
                    nk = tuple(nk)
                    new_state[nk] = new_state.get(nk, 0) + v * c
                    
        self.state = new_state
        self.remove_zero_entries()

    def apply_two_qubit_gate(self, t1: int, t2: int, matrix: List[List[complex]], controls: Optional[List[int]] = None) -> None:
        if controls:
            # similar split
            activated = QubitState(self.n_qubits)
            activated.clear()
            deactivated = QubitState(self.n_qubits)
            deactivated.clear()
            for k, v in self.state.items():
                if all(k[c] for c in controls):
                    activated.state[k] = v
                else:
                    deactivated.state[k] = v
            activated.apply_two_qubit_gate(t1, t2, matrix)
            self.state = activated.state
            for k, v in deactivated.state.items():
                self.state[k] = self.state.get(k, 0) + v
            self.remove_zero_entries()
            return
        new_state: Dict[StateKey, complex] = {}
        for k, v in self.state.items():
            i = int(k[t1]) + 2*int(k[t2])
            for row in range(4):
                nk = list(k)
                nk[t1] = bool(row & 1)
                nk[t2] = bool((row & 2) >> 1)
                nk = tuple(nk)
                new_state[nk] = new_state.get(nk, 0) + matrix[row][i] * v
        self.state = new_state
        self.remove_zero_entries()

    def reorder_index(self, old_i: int, new_i: int) -> None:
        if old_i == new_i:
            return
        if not (0 <= old_i < self.n_qubits and 0 <= new_i < self.n_qubits):
            raise IndexError("Qubit index out of range")
        new_state: Dict[StateKey, complex] = {}
        for k, v in self.state.items():
            lst = list(k)
            bit = lst.pop(old_i)
            lst.insert(new_i, bit)
            new_state[tuple(lst)] = v
        self.state = new_state

    @staticmethod
    def combine(qs1: 'QubitState', indices1: List[int], qs2: 'QubitState', indices2: List[int]) -> 'QubitState':
        # interlace sorted indices
        interlace = []
        while indices1 or indices2:
            if not indices1:
                interlace.append(False)
                indices2.pop(0)
                continue

            if not indices2:
                interlace.append(True)
                indices1.pop(0)
                continue

            if indices1[0] < indices2[0]:
                interlace.append(True)
                indices1.pop(0)
            else:
                interlace.append(False)
                indices2.pop(0)

        new_size = qs1.n_qubits + qs2.n_qubits
        new_qs = QubitState(new_size)
        new_qs.clear()

        for key1, val1 in qs1.state.items():
            for key2, val2 in qs2.state.items():
                # 1. buffer booleans della lunghezza finale
                new_key = [False] * new_size
                next_bit_new = 0
                next_bit1 = 0
                next_bit2 = 0

                for next_is_from1 in interlace:
                    if next_is_from1:
                        new_key[next_bit_new] = key1[next_bit1]
                        next_bit1 += 1
                    else:
                        new_key[next_bit_new] = key2[next_bit2]
                        next_bit2 += 1
                    next_bit_new += 1 

                new_qs.state[tuple(new_key)] = val1 * val2

        return new_qs
    
    def always_activated(self, indices: List[int]) -> bool:
        return all(all(k[i] for i in indices) for k in self.state)

    def never_activated(self, indices: List[int]) -> bool:
        return all(not all(k[i] for i in indices) for k in self.state)

class QubitStateOrTop:
    def __init__(self, variant: Optional[QubitState] = None):
        if variant is None:
            self.variant: Union[str, QubitState] = "TOP"
        else:
            self.variant = variant

    def is_top(self) -> bool:
        return self.variant == "TOP"

    def is_qubit_state(self) -> bool:
        return isinstance(self.variant, QubitState)

    def get_qubit_state(self) -> QubitState:
        if self.is_qubit_state():
            return self.variant
        raise ValueError("Not a QubitState")

    def __eq__(self, other: 'QubitStateOrTop') -> bool:
        if self.is_top() and other.is_top():
            return True
        if self.is_top() or other.is_top():
            return False
        return self.get_qubit_state() == other.get_qubit_state()

    def __str__(self) -> str:
        return "TOP" if self.is_top() else str(self.get_qubit_state())

    def __repr__(self) -> str:
        return self.__str__()
