from qiskit.circuit.classical import expr
from qiskit.circuit import Clbit
from util.BitState import BitState


class CondResult:
    def __init__(self, always_true: bool = False, always_false: bool = False, expr_node=None):
        self.always_true = always_true
        self.always_false = always_false
        self.expr = expr_node

class SimplifyCondition:
    @staticmethod
    def _bit_value(clbit: Clbit, clbit_states) -> int | None:
        st = clbit_states.get(clbit, None)
        if st == BitState.ONE:
            return 1
        if st == BitState.ZERO:
            return 0
        return None

    @classmethod
    def simplify(cls, node, clbit_states) -> CondResult:
        # Variable (Clbit)
        if isinstance(node, expr.Var):
            var = node.var
            st = clbit_states.get(var, BitState.ZERO)
            if st == BitState.ONE:
                return CondResult(always_true=True, expr_node=node)
            elif st == BitState.ZERO:
                return CondResult(always_false=True, expr_node=node)
            else:
                return CondResult(expr_node=expr.Var(var, node.type))

        # Unary operation
        if isinstance(node, expr.Unary):
            op = node.op # The operation of the node
            operand = node.operand # The operand of the node

            if op == expr.Unary.Op.BIT_NOT:
                sub = cls.simplify(operand, clbit_states)
                if sub.always_true:
                    return CondResult(always_false=True)
                if sub.always_false:
                    return CondResult(always_true=True)
                return CondResult(expr_node=expr.bit_not(sub.expr))
            else:
                # Unsupported unary operation
                return CondResult(expr_node=node)
        
        # Binary operation
        if isinstance(node, expr.Binary):
            op = node.op  # The operation of the node
            left = node.left # Left operand
            right = node.right # Right operand

            if op == expr.Binary.Op.BIT_AND:
                l = cls.simplify(left, clbit_states)
                r = cls.simplify(right, clbit_states)

                if l.always_false or r.always_false:
                    return CondResult(always_false=True)
                if l.always_true:
                    return r
                if r.always_true:
                    return l
                return CondResult(expr_node=expr.Binary(expr.Binary.Op.BIT_AND, l.expr, r.expr, type=node.type))
            elif op == expr.Binary.Op.BIT_OR:
                l = cls.simplify(left, clbit_states)
                r = cls.simplify(right, clbit_states)
                if l.always_true or r.always_true:
                    return CondResult(always_true=True)
                if l.always_false:
                    return r
                if r.always_false:
                    return l
                return CondResult(expr_node=expr.Binary(expr.Binary.Op.BIT_OR, l.expr, r.expr, type=node.type))
            elif op == expr.Binary.Op.BIT_XOR:
                l = cls.simplify(left, clbit_states)
                r = cls.simplify(right, clbit_states)
                if l.always_false:
                    return r
                if r.always_false:
                    return l
                if l.always_true:
                    if r.always_true:
                        return CondResult(always_false=True)
                    return CondResult(expr_node=expr.bit_not(r.expr))
                if r.always_true:
                    return CondResult(expr_node=expr.bit_not(l.expr))
                # TODO: check if the following is correct
                if l.expr == r.expr:
                    return CondResult(always_false=True)
                return CondResult(expr_node=expr.Binary(expr.Binary.Op.BIT_XOR, l.expr, r.expr, type=node.type))
            else:
                # Unsupported binary operation
                return CondResult(expr_node=node)

        # Unknown node type, return as it is
        return CondResult(expr_node=node)
