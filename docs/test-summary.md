# JOINT test and release verification

Verified on 2026-09-02 with Python 3.12.13 (`.venv/bin/python --version`). The final runtime
source baseline is JOINT commit `8b178116ef251582108eade660cdd33fd6b27dc0`.

## Automated checks

```bash
.venv/bin/python -W error -m pytest -m 'not regression' -q
.venv/bin/ruff check src tests
git diff --check
```

The warning-as-error non-regression suite reported **1096 passed, 4 deselected** in 35.59
seconds. Ruff reported **All checks passed!**, and `git diff --check` completed without errors.
The current provenance and pipeline-focused suite reported **175 passed** in 8.02 seconds.

The quick-start notebook was executed with the reference-data root set:

```bash
JOINT_DATA_ROOT=/Users/mac/Desktop/nanoMap/nanoMAP_figures/Figure5/5B \
.venv/bin/python - <<'PY'
import nbformat
from nbclient import NotebookClient

path = "examples/joint_quickstart.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=1200, kernel_name="python3").execute(cwd="examples")
PY
```

All four code cells executed successfully. The generated `results/` directory was removed
afterward. The committed notebook intentionally remains without cell outputs or execution counts,
so it is a clean, executable example.

## Reference regression

```bash
JOINT_DATA_ROOT=/Users/mac/Desktop/nanoMap/nanoMAP_figures/Figure5/5B \
.venv/bin/python -m pytest -m regression tests/regression/test_reference_data.py -v
```

This command reported **4 passed** in 108.73 seconds, with no failures or skips. The notebook
reproduction baselines were:

| Config | Laser observations | Cell observations |
| --- | ---: | ---: |
| `configs/d2.yaml` | 1,480 | 2,936 |
| `configs/d8.yaml` | 1,640 | 4,986 |

The raw imZML pixel-count checks also passed: d2 = 1,760 and d8 = 1,640.

## Build and installed-package checks

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
PIP_CONFIG_FILE=/dev/null .venv/bin/python -m build
```

The isolated build produced `joint_msi-0.1.0.tar.gz` and
`joint_msi-0.1.0-py3-none-any.whl`.

## Final dependency-resolution verification

The dependency-resolving final-wheel install was rechecked in a fresh Python 3.12 virtual
environment after adding `numba>=0.60,<0.62` to the runtime metadata:

```bash
.venv/bin/python -m venv "$FINAL_WHEEL_ENV/venv"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
PIP_CONFIG_FILE=/dev/null "$FINAL_WHEEL_ENV/venv/bin/python" -m pip install --force-reinstall \
  dist/joint_msi-0.1.0-py3-none-any.whl
"$FINAL_WHEEL_ENV/venv/bin/python" -c "import joint; print(joint.__version__)"
"$FINAL_WHEEL_ENV/venv/bin/python" -m pip check
PATH="$FINAL_WHEEL_ENV/venv/bin:$PATH" joint --help
PATH="$FINAL_WHEEL_ENV/venv/bin:$PATH" joint run --help
```

Dependency resolution selected `numba 0.61.2`, `llvmlite 0.44.0`, and `numpy 2.2.6`, all from
available Python 3.12 wheels. The install completed; import printed `0.1.0`, `pip check`
reported **No broken requirements found**, and both CLI help commands exited zero. The installed
package also exposed all seven published reference resources through `importlib.resources`.
The verified wheel SHA-256 was
`ad3e3ecee8304ab502c457aa0fefb7cecddf9ce6ce86b6ef44d9eeeae42dd676`.
