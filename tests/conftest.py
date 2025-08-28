# tests/conftest.py
import os, sys
# add project root to sys.path so `import src...` works from tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
