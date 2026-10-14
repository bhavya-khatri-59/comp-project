import ast


class ASTOptimizer(ast.NodeTransformer):
    """Performs constant folding and algebraic simplification."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)

        # Constant Folding: e.g., 60 * 60 -> 3600
        if isinstance(node.left, ast.Constant) and isinstance(
            node.right, ast.Constant
        ):
            left_val, right_val = node.left.value, node.right.value
            try:
                if isinstance(node.op, ast.Add):
                    return ast.Constant(value=left_val + right_val)
                elif isinstance(node.op, ast.Sub):
                    return ast.Constant(value=left_val - right_val)
                elif isinstance(node.op, ast.Mult):
                    return ast.Constant(value=left_val * right_val)
                elif isinstance(node.op, ast.Div):
                    return ast.Constant(value=left_val / right_val)
                elif isinstance(node.op, ast.FloorDiv):
                    return ast.Constant(value=left_val // right_val)
                elif isinstance(node.op, ast.Mod):
                    return ast.Constant(value=left_val % right_val)
            except ZeroDivisionError:
                return node

        # Algebraic Simplifications: e.g., x * 1 -> x, x + 0 -> x, x * 0 -> 0
        if isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant):
                if node.right.value == 1:
                    return node.left
                elif node.right.value == 0:
                    return ast.Constant(value=0)
            if isinstance(node.left, ast.Constant):
                if node.left.value == 1:
                    return node.right
                elif node.left.value == 0:
                    return ast.Constant(value=0)

        if isinstance(node.op, ast.Add):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                return node.left
            if isinstance(node.left, ast.Constant) and node.left.value == 0:
                return node.right

        return node


def optimize_source(source_code: str) -> str:
    """Parses code, applies AST optimization passes, and unparses back to code."""
    tree = ast.parse(source_code)
    optimizer = ASTOptimizer()
    optimized_tree = optimizer.visit(tree)
    ast.fix_missing_locations(optimized_tree)
    return ast.unparse(optimized_tree)