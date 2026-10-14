import ast
import streamlit as st
import graphviz

from compiler.ast_analyzer import optimize_source
from compiler.cfg_builder import CFGBuilder
from compiler.leak_detector import LeakDetector
from agent.patcher import generate_and_verify_patch

st.set_page_config(
    page_title="Compiler & Static Analysis Engine",
    page_icon="🛡️",
    layout="wide"
)

st.title("Static Resource Leak & AST Optimizer")
st.caption("A deterministic compiler static analyzer paired with an AST-verified Groq refactoring engine.")

SAMPLE_CODE = '''def process_user_data(filepath, mode):
    f = open(filepath, "r")
    multiplier = 60 * 60 * 24
    if mode == "fast":
        return f.readline() * 1
    
    data = f.read()
    f.close()
    return data
'''

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Source Code Input")
    source_input = st.text_area(
        "Enter Python code to analyze:",
        value=SAMPLE_CODE,
        height=260
    )
    analyze_btn = st.button("🚀 Run Compiler Analysis & Optimization", type="primary", use_container_width=True)

if analyze_btn:
    # --- Pass 1: Deterministic AST Optimization ---
    try:
        optimized_code = optimize_source(source_input)
    except SyntaxError as e:
        st.error(f"❌ Syntax Error in Source Code (Line {e.lineno}): {e.msg}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error during AST Optimization: {str(e)}")
        st.stop()

    # --- Pass 2: Control Flow Graph Generation ---
    tree = ast.parse(optimized_code)
    cfg_builder = CFGBuilder()
    cfg = cfg_builder.build_cfg(tree)

    # --- Pass 3: Static Data-Flow Leak Analysis ---
    detector = LeakDetector(cfg)
    leaks = detector.detect_leaks()

    # Collect nodes on leaking paths to highlight in graph
    leaky_nodes = set()
    for leak in leaks:
        for node_id in leak.get("execution_path", []):
            leaky_nodes.add(node_id)

    with col2:
        st.subheader("2. Compiler Analysis & Diagnostics")
        st.markdown("**AST Optimized Code (Constant Folding & Identity Simplification):**")
        st.code(optimized_code, language="python")

        if leaks:
            st.error(f"⚠️ {len(leaks)} Resource Leak Path(s) Detected!")
            for leak in leaks:
                st.warning(f"• **{leak['variable']}** ({leak['resource_type']}) at line {leak['allocated_line']}:\n{leak['message']}")
        else:
            st.success("✅ Clean Execution: No unclosed resource handles detected across any execution path.")

    st.divider()

    # --- Visual CFG Section ---
    st.subheader("3. Control Flow Graph (CFG)")
    dot = graphviz.Digraph(comment="CFG", node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica"})
    
    for node, data in cfg.nodes(data=True):
        label = data.get("label", str(node))
        if node in leaky_nodes:
            # Highlight blocks involved in the resource leak
            dot.node(str(node), label, fillcolor="#ffcccc", color="#cc0000", fontcolor="#800000")
        else:
            dot.node(str(node), label, fillcolor="#e6f2ff", color="#0066cc", fontcolor="#002b66")

    for u, v, data in cfg.edges(data=True):
        edge_label = data.get("label", "")
        if u in leaky_nodes and v in leaky_nodes:
            dot.edge(str(u), str(v), label=edge_label, color="#cc0000", penwidth="2.0")
        else:
            dot.edge(str(u), str(v), label=edge_label, color="#666666")

    st.graphviz_chart(dot)

    # --- Pass 4: Groq Auto-Patcher & Compiler Verification ---
    if leaks:
        st.divider()
        st.subheader("4. 🤖 Groq LLM Auto-Patcher & Re-Verification Loop")
        
        with st.spinner("Generating semantic patch via Groq and verifying AST..."):
            fixed_code, is_verified, status_msg = generate_and_verify_patch(optimized_code, leaks)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Generated Patch:**")
            st.code(fixed_code, language="python")
        with c2:
            st.markdown("**Compiler Verification Result:**")
            if is_verified:
                st.success(f"**Passed:** {status_msg}")
                st.info("The patched code parsed into a valid AST with 0 unclosed handles remaining.")
            else:
                st.error(f"**Failed:** {status_msg}")