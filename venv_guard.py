"""
Garante que os scripts do projeto rodem num Python com as dependências instaladas.

No Windows há vários Pythons nesta máquina (3.14 do sistema, .venv, .venv312...)
e só alguns têm TensorFlow. Se o script for iniciado por um Python sem os módulos
exigidos — botão Run do VS Code, `python main.py` no terminal — ele é reiniciado
automaticamente com o .venv312, em vez de falhar com "No module named ...".
Fora do Windows (WSL, run_wsl.sh) não faz nada.
"""
import importlib.util
import os
import subprocess
import sys

VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv312", "Scripts", "python.exe")


def ensure(*modules):
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing or sys.platform != "win32":
        return
    here = os.path.normcase(os.path.abspath(sys.executable))
    if not os.path.exists(VENV_PYTHON) or here == os.path.normcase(VENV_PYTHON):
        sys.exit(
            f"Faltam os módulos {', '.join(missing)} em {sys.executable}.\n"
            f"Instale com: {VENV_PYTHON} -m pip install -r requirements.txt"
        )

    print(f"[faltam {', '.join(missing)} em {sys.executable}; reiniciando com o .venv312]", flush=True)
    proc = subprocess.Popen([VENV_PYTHON, os.path.abspath(sys.argv[0]), *sys.argv[1:]])
    while True:
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:
            continue   # o Ctrl+C também chega ao processo filho, que encerra sozinho
