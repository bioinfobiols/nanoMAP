# JOINT 合成测试数据集与 README 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 JOINT 增加可重复生成并直接提交的小型真实格式测试数据，同时把 README 扩展为从安装到 CLI/Python API/真实数据的完整中文使用手册。

**Architecture:** 将数据生成逻辑集中在 `tests/data/synthetic_demo/generate_dataset.py`，以同一组确定性光谱、空间坐标和标签同时写出 h5ad 与 imzML/ibd。配置文件只引用 fixture 目录内的相对路径，测试通过 `load_config` 和实际 I/O 验证两条输入链路；README 只描述已存在的命令和文件，不把真实 d2/d8 数据伪装成 fixture。

**Tech Stack:** Python 3.12-compatible Python 3.11+、NumPy、pandas、AnnData、Pillow、PyYAML、pyimzML、pytest、Typer。

## Global Constraints

- 数据规模固定为 32×32 像素，共 1024 个 MSI 观测。
- 数据必须同时提供原始 imzML/ibd、预处理 h5ad、激光图像和细胞分割标签。
- 使用固定 seed；生成器重复执行必须产生相同的数值内容和元数据。
- 生成器默认拒绝覆盖非空输出目录，只有 `--overwrite` 允许替换。
- `tests/data/synthetic_demo/` 中提交一份生成好的数据，且保留可重建脚本。
- 不提交 `.venv`、缓存、pipeline `results/` 或构建临时目录。
- 不修改现有 d2/d8 配置的默认路径、真实回归基线和 CLI 接口。
- 所有新增测试使用 `.venv/bin/python` 执行；完成前运行非回归测试、Ruff 和 `git diff --check`。

---

### Task 1: 实现确定性的合成数据生成器

**Files:**
- Create: `tests/data/synthetic_demo/generate_dataset.py`
- Create: `tests/data/synthetic_demo/__init__.py`
- Test: `tests/test_synthetic_dataset.py`

**Interfaces:**
- Produces `generate_dataset(output_dir: str | Path, *, seed: int = 20260912, overwrite: bool = False) -> Path`.
- Produces `build_synthetic_arrays(seed: int = 20260912) -> dict[str, np.ndarray]` for unit-level deterministic checks.
- Produces files named `synthetic_demo.h5ad`, `synthetic_demo.imzML`, `synthetic_demo.ibd`, `laser_image.png`, `cell_segmentation.npy`, and `manifest.json` under the output directory.

- [ ] **Step 1: Write failing tests for deterministic arrays and output policy**

  Add tests that call the not-yet-existing functions:

  ```python
  def test_build_synthetic_arrays_is_deterministic():
      first = build_synthetic_arrays(seed=20260912)
      second = build_synthetic_arrays(seed=20260912)
      assert first.keys() == second.keys()
      for name in first:
          np.testing.assert_array_equal(first[name], second[name])

  def test_generate_dataset_rejects_nonempty_output_without_overwrite(tmp_path):
      output = tmp_path / "synthetic_demo"
      output.mkdir()
      (output / "keep.txt").write_text("keep")
      with pytest.raises(FileExistsError, match="overwrite"):
          generate_dataset(output)

  def test_generate_dataset_writes_expected_files(tmp_path):
      output = generate_dataset(tmp_path / "synthetic_demo")
      assert (output / "synthetic_demo.h5ad").is_file()
      assert (output / "synthetic_demo.imzML").is_file()
      assert (output / "synthetic_demo.ibd").is_file()
      assert (output / "laser_image.png").is_file()
      assert (output / "cell_segmentation.npy").is_file()
      assert (output / "manifest.json").is_file()
  ```

- [ ] **Step 2: Run the focused tests to verify RED**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_synthetic_dataset.py -q
  ```

  Expected: collection/import failure because `tests.data.synthetic_demo.generate_dataset` does not yet exist.

- [ ] **Step 3: Implement deterministic array construction**

  Use `np.random.default_rng(seed)` and return these exact keys and shapes:

  ```python
  {
      "coordinates": shape (1024, 2), dtype int32, values 1..32 in row-major order,
      "mz": shape (10,), dtype float64, strictly increasing positive values,
      "intensities": shape (1024, 10), dtype float32, finite non-negative values,
      "laser_image": shape (32, 32), dtype uint8,
      "cell_labels": shape (32, 32), dtype int32, labels in {0, 1, 2, 3, 4},
  }
  ```

  Generate the 10 m/z values as deterministic values beginning at 100.0 Da with fixed 25.0 Da spacing. Build four rectangular cell regions using the four 16×16 quadrants, reserve a one-pixel background border, and add a deterministic spatial gradient plus seed-controlled noise to intensities. Build the laser image from a background value of 10 and bright horizontal/vertical marker bands, then clip to `uint8`.

- [ ] **Step 4: Implement h5ad and image/label writers**

  Construct `AnnData` with `X` as a CSR sparse matrix, `obs["dataset"] == "synthetic_demo"`, `var["mz"] == mz`, and `obsm["spatial"] == coordinates`; write it through the existing `atomic_write_h5ad`. Save the image as an 8-bit grayscale PNG with Pillow and labels with `np.save`.

- [ ] **Step 5: Implement imzML/ibd writing**

  Use `pyimzml.ImzMLWriter.ImzMLWriter` with `mz_dtype=np.float64`, `intensity_dtype=np.float32`, `spec_type="centroid"`. For each row-major coordinate `(x, y)`, call `addSpectrum(mz, intensities[index], (int(x), int(y), 1))`, then close the writer only after all 1024 spectra are added. Write into a temporary directory and atomically promote the completed pair into the destination so a failed writer cannot leave a partial fixture.

- [ ] **Step 6: Write manifest and generator CLI**

  Record `seed`, `shape`, `pixel_count`, `feature_count`, `m/z`, `cell_labels`, `files`, and SHA-256 hashes in sorted JSON. Implement:

  ```bash
  python tests/data/synthetic_demo/generate_dataset.py \
    --output-dir tests/data/synthetic_demo \
    --seed 20260912 \
    --overwrite
  ```

  The CLI must use `argparse`, default to the script directory as the output directory, and return a non-zero exit code with a readable error for invalid seed, missing parent, or overwrite refusal.

- [ ] **Step 7: Run focused generator tests and commit**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_synthetic_dataset.py -q
  .venv/bin/ruff check tests/data/synthetic_demo/generate_dataset.py tests/test_synthetic_dataset.py
  ```

  Expected: all focused tests pass and Ruff reports no errors. Commit as `feat: add synthetic JOINT dataset generator`.

### Task 2: Generate fixture files and configs

**Files:**
- Create: `tests/data/synthetic_demo/README.md`
- Create: `tests/data/synthetic_demo/configs/synthetic-h5ad.yaml`
- Create: `tests/data/synthetic_demo/configs/synthetic-imzml.yaml`
- Generate: `tests/data/synthetic_demo/synthetic_demo.h5ad`
- Generate: `tests/data/synthetic_demo/synthetic_demo.imzML`
- Generate: `tests/data/synthetic_demo/synthetic_demo.ibd`
- Generate: `tests/data/synthetic_demo/laser_image.png`
- Generate: `tests/data/synthetic_demo/cell_segmentation.npy`
- Generate: `tests/data/synthetic_demo/manifest.json`
- Test: `tests/test_synthetic_dataset.py`

**Interfaces:**
- `synthetic-h5ad.yaml` uses `input.h5ad: ../synthetic_demo.h5ad` and the shared image/label files.
- `synthetic-imzml.yaml` uses `input.imzml: ../synthetic_demo.imzML` and the same image/label files.
- Both configs use `project.output_dir: ../../../results/synthetic-*`, `laser_segmentation.method: simulated`, `registration.orientation: identity`, and `quantification.method: specificity_filtered`.

- [ ] **Step 1: Add config and fixture metadata tests**

  Extend `tests/test_synthetic_dataset.py` with:

  ```python
  def test_fixture_shapes_and_manifest():
      root = Path(__file__).parent / "data" / "synthetic_demo"
      adata = anndata.read_h5ad(root / "synthetic_demo.h5ad")
      labels = np.load(root / "cell_segmentation.npy")
      assert adata.shape == (1024, 10)
      assert adata.obsm["spatial"].shape == (1024, 2)
      assert labels.shape == (32, 32)
      assert set(np.unique(labels)) == {0, 1, 2, 3, 4}
  ```

- [ ] **Step 2: Run the new test to verify it fails before fixtures exist**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_synthetic_dataset.py::test_fixture_shapes_and_manifest -q
  ```

  Expected: FAIL because the checked-in `.h5ad` and label files do not yet exist.

- [ ] **Step 3: Add the two YAML configs**

  Use paths relative to each config file. Set the prebuilt configuration to `preprocessing.normalization: none` and the raw configuration to `preprocessing.normalization: rms`, `preprocessing.peak_tolerance: 10`, and `preprocessing.tolerance_unit: da`. Set `analysis.cosg.enabled: false`, `analysis.cnmf.enabled: false`, `qc.enabled: false`, `annotation.enabled: false`, and `trajectory.enabled: false` so the demo does not require optional packages.

- [ ] **Step 4: Run the generator and write fixture README**

  Run:

  ```bash
  .venv/bin/python tests/data/synthetic_demo/generate_dataset.py \
    --output-dir tests/data/synthetic_demo \
    --seed 20260912 \
    --overwrite
  ```

  Write `tests/data/synthetic_demo/README.md` with the exact regeneration command, file table, seed, expected shapes, and a note that the fixture is synthetic and not a scientific benchmark.

- [ ] **Step 5: Verify both I/O paths and commit**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_synthetic_dataset.py -q
  .venv/bin/python - <<'PY'
  from pathlib import Path
  from joint.io import read_imzml
  root = Path("tests/data/synthetic_demo")
  adata = read_imzml(root / "synthetic_demo.imzML", tolerance=0.01, unit="da")
  assert adata.shape == (1024, 10)
  print(adata.shape)
  PY
  ```

  Expected: focused tests pass and the imzML reader prints `(1024, 10)`. Commit as `test: add generated JOINT demo fixtures`.

### Task 3: Add end-to-end synthetic smoke coverage

**Files:**
- Modify: `tests/test_synthetic_dataset.py`
- Modify: `tests/data/synthetic_demo/configs/synthetic-h5ad.yaml`
- Modify: `tests/data/synthetic_demo/configs/synthetic-imzml.yaml`

**Interfaces:**
- Tests consume `JointPipeline.from_config`, `read_imzml`, and both checked-in configs.
- Tests write only under pytest `tmp_path`; no repository `results/` directory is allowed.

- [ ] **Step 1: Add failing pipeline smoke assertions**

  Add a test that copies a selected config to `tmp_path`, rewrites its output directory to a temporary path, calls `JointPipeline.from_config(...).run(resume=False, overwrite=False)`, and asserts `result.adata.shape[0] == 4` and these artifacts exist: `manifest.json`, `logs/joint.log`, `preprocessing/msi.h5ad`, `segmentation/laser_labels.npy`, `registration/laser.h5ad`, and `quantification/cells.h5ad`.

- [ ] **Step 2: Run the smoke test and record the first failure**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_synthetic_dataset.py::test_synthetic_pipeline_smoke -q
  ```

  Expected: FAIL until the configs and fixture paths are wired correctly.

- [ ] **Step 3: Adjust only config-relative paths and synthetic segmentation parameters**

  Keep the generator data unchanged. Make the simulated laser configuration use the 32×32 image dimensions, a disk radius of 2, a deterministic row count, and `quantification.boundary_margin: 0` so the four cell regions remain measurable.

- [ ] **Step 4: Run both config smoke tests**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_synthetic_dataset.py::test_synthetic_pipeline_smoke -q
  ```

  Expected: both h5ad and imzML configurations complete with no optional dependency installation.

- [ ] **Step 5: Commit**

  Commit as `test: cover synthetic JOINT end-to-end workflow`.

### Task 4: Expand the Chinese README

**Files:**
- Modify: `README.md`
- Test: `tests/test_synthetic_dataset.py`

**Interfaces:**
- README commands must reference `tests/data/synthetic_demo` and the two config files exactly as committed.
- Existing d2/d8 commands and API examples remain valid.

- [ ] **Step 1: Add documentation contract tests**

  Assert README contains the exact strings `generate_dataset.py`, `synthetic-h5ad.yaml`, `synthetic-imzml.yaml`, `joint run`, `JointPipeline.from_config`, `--resume`, `--overwrite`, `JOINT_DATA_ROOT`, and `logs/joint.log`.

- [ ] **Step 2: Run the contract test to verify RED**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_synthetic_dataset.py::test_readme_documents_synthetic_and_real_workflows -q
  ```

  Expected: FAIL because the current README does not yet contain the complete synthetic workflow and all requested output/error guidance.

- [ ] **Step 3: Rewrite README sections**

  Keep the existing installation and real d2/d8 sections, then add sections in this order: 软件定位、安装、最小测试数据、h5ad CLI、imzML CLI、Python API、分阶段执行、输出结果、配置字段、真实 d2/d8 数据、常见错误、开发者测试。 Show commands with repository-relative paths and explicitly state that `joint run` writes checkpointed results and can resume.

- [ ] **Step 4: Verify rendered Markdown content and commit**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_synthetic_dataset.py::test_readme_documents_synthetic_and_real_workflows -q
  git diff --check
  ```

  Expected: contract test passes and diff check is clean. Commit as `docs: document JOINT synthetic and real workflows`.

### Task 5: Final verification and release evidence

**Files:**
- Modify: `docs/test-summary.md`
- Test: `tests/test_synthetic_dataset.py`

**Interfaces:**
- The final test summary records the synthetic fixture command, both config smoke results, and the full test command.

- [ ] **Step 1: Run the complete synthetic and existing verification gates**

  Run:

  ```bash
  .venv/bin/python -W error -m pytest tests/test_synthetic_dataset.py -q
  .venv/bin/python -W error -m pytest -m 'not regression' -q
  .venv/bin/ruff check src tests
  git diff --check
  ```

  Expected: all synthetic tests and all non-regression tests pass; Ruff and diff check are clean.

- [ ] **Step 2: Execute the quickstart notebook and clean outputs**

  Run the existing `examples/joint_quickstart.ipynb` verification with `JOINT_DATA_ROOT` and remove generated `results/` afterward, preserving null execution counts and empty outputs in the committed notebook.

- [ ] **Step 3: Build and inspect release artifacts**

  Run `.venv/bin/python -m build` and assert the wheel/sdist contain both synthetic configs, generator script, fixture README, and the generated input files. Do not add build outputs to git.

- [ ] **Step 4: Update test summary and commit**

  Record fixture dimensions, seed, synthetic smoke results, full test counts, notebook verification, and build status in `docs/test-summary.md`. Commit as `docs: record synthetic fixture verification`.

- [ ] **Step 5: Review final worktree**

  Run:

  ```bash
  git status --short
  git log --oneline -8
  ```

  Expected: only intentional source, docs, tests, configs, and fixture files are tracked; no cache, `results/`, `.venv`, or build output is tracked.
