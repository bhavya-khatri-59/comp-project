import ast
import networkx as nx
from typing import List

class BasicBlock:
    def __init__(self, block_id: int, label: str):
        self.block_id = block_id
        self.label = label
        self.statements: List[ast.AST] = []

    def __str__(self):
        return f"Block {self.block_id}: {self.label}"

class CFGBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.block_counter = 0

    def new_block(self, label: str = "") -> int:
        bid = self.block_counter
        self.block_counter += 1
        block = BasicBlock(bid, label)
        self.graph.add_node(bid, block=block, label=f"B{bid}: {label}")
        return bid

    def build_cfg(self, tree: ast.AST) -> nx.DiGraph:
        entry = self.new_block("Entry")
        exit_block = self.new_block("Exit")

        # If top-level contains a FunctionDef, traverse into the function's body
        statements = []
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.FunctionDef):
                statements.extend(node.body)
            else:
                statements.append(node)

        current = entry
        for stmt in statements:
            current = self._process_statement(stmt, current, exit_block)

        if current != exit_block and not self.graph.has_edge(current, exit_block):
            self.graph.add_edge(current, exit_block)

        return self.graph

    def _process_statement(self, node: ast.AST, current_block: int, exit_block: int) -> int:
        if isinstance(node, ast.If):
            cond_label = f"if: {ast.unparse(node.test)}"
            cond_block = self.new_block(cond_label)
            self.graph.add_edge(current_block, cond_block)

            # True / Then branch
            then_block = self.new_block("then")
            self.graph.add_edge(cond_block, then_block, label="True")
            curr_then = then_block
            for stmt in node.body:
                curr_then = self._process_statement(stmt, curr_then, exit_block)

            # False / Else branch
            else_block = self.new_block("else")
            self.graph.add_edge(cond_block, else_block, label="False")
            curr_else = else_block
            for stmt in node.orelse:
                curr_else = self._process_statement(stmt, curr_else, exit_block)

            # Join block if branches do not terminate at Exit
            join_block = self.new_block("join_if")
            if curr_then != exit_block:
                self.graph.add_edge(curr_then, join_block)
            if curr_else != exit_block:
                self.graph.add_edge(curr_else, join_block)

            return join_block

        elif isinstance(node, ast.Return):
            ret_label = f"return: {ast.unparse(node.value) if node.value else ''}"
            ret_block = self.new_block(ret_label)
            self.graph.nodes[ret_block]["block"].statements.append(node)
            self.graph.add_edge(current_block, ret_block)
            self.graph.add_edge(ret_block, exit_block)
            return exit_block

        # ... existing ast.If and ast.Return logic ...

        elif isinstance(node, ast.With):
            # Create a block for the 'with' statement itself
            stmt_label = f"with: {ast.unparse(node.items[0])}"
            with_block = self.new_block(stmt_label)
            self.graph.nodes[with_block]["block"].statements.append(node)
            self.graph.add_edge(current_block, with_block)
            
            # Process the body of the with block
            curr = with_block
            for stmt in node.body:
                curr = self._process_statement(stmt, curr, exit_block)
            return curr

        elif isinstance(node, ast.Try):
            # 1. Create a dedicated block for the finally statements
            finally_start = self.new_block("finally")
            
            # 2. Process the try body. 
            curr = current_block
            for stmt in node.body:
                curr = self._process_statement(stmt, curr, finally_start)
                
            # 3. Connect normal fall-through execution to the finally block
            if curr != finally_start and not self.graph.has_edge(curr, finally_start):
                self.graph.add_edge(curr, finally_start)
                
            # 4. Now process the finally body (which will eventually route to the real exit)
            curr_fin = finally_start
            for stmt in node.finalbody:
                curr_fin = self._process_statement(stmt, curr_fin, exit_block)
                
            return curr_fin

        elif isinstance(node, (ast.For, ast.While)):
            # Loop header
            loop_label = f"loop: {ast.unparse(node).split(':')[0]}"
            loop_block = self.new_block(loop_label)
            self.graph.add_edge(current_block, loop_block)
            
            # Loop body
            curr_body = loop_block
            for stmt in node.body:
                curr_body = self._process_statement(stmt, curr_body, exit_block)
            
            # Edge back to loop header
            if curr_body != exit_block:
                self.graph.add_edge(curr_body, loop_block)
                
            # Exit loop path
            loop_exit = self.new_block("loop_exit")
            self.graph.add_edge(loop_block, loop_exit)
            return loop_exit
        elif isinstance(node, ast.Try):
            # 1. Create a dedicated block for the finally statements
            finally_start = self.new_block("finally")
            
            # 2. Process the try body. 
            # CRITICAL FIX: We pass `finally_start` as the exit_block parameter! 
            # This forces any early `ast.Return` inside the try block to route 
            # through the finally block before actually exiting.
            curr = current_block
            for stmt in node.body:
                curr = self._process_statement(stmt, curr, finally_start)
                
            # 3. Connect normal fall-through execution to the finally block
            if curr != finally_start and not self.graph.has_edge(curr, finally_start):
                self.graph.add_edge(curr, finally_start)
                
            # 4. Now process the finally body (which will eventually route to the real exit)
            curr_fin = finally_start
            for stmt in node.finalbody:
                curr_fin = self._process_statement(stmt, curr_fin, exit_block)
                
            return curr_fin

        else:
            stmt_label = f"L{getattr(node, 'lineno', '?')}: {ast.unparse(node)}"
            stmt_block = self.new_block(stmt_label)
            self.graph.nodes[stmt_block]["block"].statements.append(node)
            self.graph.add_edge(current_block, stmt_block)
            return stmt_block