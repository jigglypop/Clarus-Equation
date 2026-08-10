from pathlib import Path
import numpy as np
from reality_stone.clarus.folding_surface_3d import write_obj


def test_obj_export_has_vertices_and_faces(tmp_path: Path) -> None:
    path = tmp_path / "surface.obj"
    write_obj(np.zeros((3, 3)), path)
    text = path.read_text(encoding="ascii")
    assert text.count("v ") == 9
    assert text.count("f ") == 8
