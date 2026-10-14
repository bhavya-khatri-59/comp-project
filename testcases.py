"""
Test Cases
Copy and paste individual functions into the Streamlit UI to test the compiler pipeline.
"""

# =====================================================================
# TEST CASE 1: The Classic Early Return Leak (Branch Leak)
# =====================================================================
# WHAT IT TESTS: 
# 1. AST: Folds `60 * 60` to `3600` and simplifies `* 1`.
# 2. CFG: Splits into a True/False branch.
# 3. LEAK DETECTOR: Flags that 'f' is leaked ONLY on the early return path.
# 4. LLM: Will likely fix this by moving the file read into a `with open(...)` block.
def test_early_return_leak(filepath, fast_mode=True):
    f = open(filepath, "r")
    timeout_sec = 60 * 60
    
    if fast_mode:
        return f.readline() * 1
    
    data = f.read()
    f.close()
    return data


# =====================================================================
# TEST CASE 2: The Partial Leak (Multiple Resources)
# =====================================================================
# WHAT IT TESTS:
# 1. AST: Folds `500 + 500` to `1000`.
# 2. CFG: Linear execution block.
# 3. LEAK DETECTOR: Accurately distinguishes between handles. It will see that 
#    'target_db' is closed safely, but 'source_db' is completely forgotten.
#    It will output exactly 1 leak for 'source_db'.
def test_partial_resource_leak(src_url, dst_url):
    source_db = connect(src_url)
    target_db = connect(dst_url)
    
    batch_size = 500 + 500
    data = source_db.read(batch_size)
    target_db.write(data)
    
    target_db.disconnect()
    # BUG: source_db is never disconnected!
    return True


# =====================================================================
# TEST CASE 3: Clean Execution (Negative Test)
# =====================================================================
# WHAT IT TESTS:
# 1. AST: Removes the redundant `+ 0`.
# 2. CFG & LEAK DETECTOR: Proves that the compiler doesn't throw false positives. 
#    The resource is safely opened and closed on the only execution path.
# 3. OVERALL: Should show a green "Clean Execution" banner. The LLM will be skipped.
def test_clean_execution(log_path):
    log_file = open(log_path, "a")
    offset = 100 + 0
    
    log_file.write("Processing started at offset...")
    
    log_file.close()
    return True


# =====================================================================
# TEST CASE 4: The While-Loop Traversal
# =====================================================================
# WHAT IT TESTS:
# 1. AST: Simplifies `1 * 1024` to `1024`.
# 2. CFG: Proves your compiler can draw circular graph edges (loop header -> body -> header).
# 3. LEAK DETECTOR: Navigates the loop and realizes that when the loop exits, 
#    'client_sock' is never closed before the function returns.
def test_while_loop_leak(host, port):
    client_sock = socket(host, port)
    buffer = 1 * 1024
    retries = 0
    
    while retries < 3:
        status = client_sock.ping()
        if status == "OK":
            return True
        retries = retries + 1
        
    # BUG: If the loop finishes (retries == 3), the socket is never closed!
    return False


# =====================================================================
# TEST CASE 5: The "Try-Finally" CFG Routing Test
# =====================================================================
# WHAT IT TESTS:
# 1. CFG: This proves the custom fix we applied for `ast.Try` nodes.
#    Even though there is a `return` inside the try block, the CFG will successfully 
#    route the execution path through the `finally` block before exiting.
# 2. LEAK DETECTOR: Because the CFG routes through `finally`, the detector sees 
#    `mem_ptr.free()` and correctly marks this as a CLEAN execution (0 leaks).
def test_try_finally_routing():
    mem_ptr = malloc(2048)
    
    try:
        data = mem_ptr.read()
        return data  # CFG should route this to 'finally', not directly to exit
    finally:
        mem_ptr.free()