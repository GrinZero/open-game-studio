"""CPU-only migration checks; never import torch/bpy or connect to a GPU runtime."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if sys.version_info < (3, 11, 8):
    raise SystemExit("Monster deployment tooling requires Python 3.11.8 or later")

for directory in ("worker", "scripts", "integration", "tests", "evals", "skills"):
    for file in (ROOT / directory).rglob("*.py"):
        if "__pycache__" not in file.parts:
            compile(file.read_text(), str(file), "exec")
subprocess.run(["bash", "-n", str(ROOT / "scripts/bootstrap-skintokens.sh")], check=True)
notebook = json.loads((ROOT / "notebooks/colab.ipynb").read_text())
assert notebook["nbformat"] == 4
for cell in notebook["cells"]:
    source = "".join(cell["source"])
    assert "FILES_B64" not in source and "BOOTSTRAP_B64" not in source
    assert "base64" not in source
    if cell["cell_type"] == "code":
        assert len(source) < 5000, "Keep Notebook cells as small deployment entry points"
        compile(source, "colab.ipynb", "exec")
        assert cell["outputs"] == [] and cell["execution_count"] is None

tree = ast.parse((ROOT / "worker/colab_g4_worker_server.py").read_text())
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
protocol = json.loads((ROOT / "contracts/protocol.json").read_text())
for endpoint, arguments in protocol["endpoints"].items():
    assert [a.arg for a in functions["api_" + endpoint].args.args] == arguments, endpoint
registered = {keyword.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
              for keyword in node.keywords if keyword.arg == "api_name"}
assert registered == set(protocol["endpoints"]), "Protocol must describe the exposed Gradio endpoints"

result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], cwd=ROOT)
if result.returncode:
    raise SystemExit(result.returncode)

formal_gate_res = subprocess.run(
    [sys.executable, str(ROOT / "skills/monster-workflow/scripts/sop_formal_gate.py"), "self-test"],
    cwd=ROOT,
    capture_output=True,
    text=True,
)
if formal_gate_res.returncode:
    sys.stderr.write(formal_gate_res.stdout + formal_gate_res.stderr)
    raise SystemExit(formal_gate_res.returncode)
formal_report = json.loads(formal_gate_res.stdout)
assert formal_report.get("passed") is True, "Formal verification gate must pass"
print(
    "Monster syntax, Notebook, API contract, CPU-only migration checks, and Formal Verification Gate (P1-P4) passed."
)
