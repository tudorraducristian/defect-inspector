from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).parents[2] / "training" / "train.ipynb"


def read_notebook():
    return nbformat.read(str(NOTEBOOK), as_version=4)


def test_notebook_is_valid_and_pinned():
    nb = read_notebook()
    nbformat.validate(nb)
    text = "".join(c.source for c in nb.cells)
    assert "fcab5146f84ae17597044ad5ddf1656ccf805401" in text and "--require-gpu" in text
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


def test_notebook_checks_the_teacher_size():
    assert "10779715" in "".join(c.source for c in read_notebook().cells)


def test_notebook_never_trains_on_a_half_extracted_archive():
    """Stopped during extraction, the final folder must not exist on the next Run all."""
    text = "".join(c.source for c in read_notebook().cells)
    assert "shutil.unpack_archive(str(archive), str(partial))" in text
    assert "os.replace(partial, extracted)" in text
    assert "shutil.unpack_archive(str(archive), str(extracted))" not in text


def test_notebook_passes_the_station_when_the_dataset_has_one():
    text = "".join(c.source for c in read_notebook().cells)
    assert 'DATASET_ROOT / "station.json"' in text and "{station_flag}" in text


def without_shell_commands(source):
    """Replace IPython's ! lines with pass, keeping the indentation of the blocks."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        lines.append(line[:len(line) - len(stripped)] + "pass" if stripped.startswith("!") else line)
    return "\n".join(lines)


def test_notebook_code_cells_compile_once_shell_lines_are_replaced():
    for cell in read_notebook().cells:
        if cell.cell_type == "code":
            compile(without_shell_commands(cell.source), "cell", "exec")
