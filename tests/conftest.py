import sys
from pathlib import Path

# Make tests/functions.py importable from every test module, including those in
# subdirectories -- pytest only puts each test file's own directory on sys.path.
sys.path.insert(0, str(Path(__file__).parent))
