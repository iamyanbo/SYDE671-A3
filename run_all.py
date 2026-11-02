from pathlib import Path
import os
import runpy

os.environ.setdefault("MPLBACKEND", "Agg")
from matplotlib import pyplot as plt

plt.show = lambda: None
runpy.run_path(str(Path(__file__).resolve().parent / "A3.py"), run_name="__main__")
