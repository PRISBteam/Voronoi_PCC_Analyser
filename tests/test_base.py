import pytest
from matgen import base

def test_Vertex():
    """
    """
    v = base.Vertex(id=42, x=0.34, y=0.58, z=0.89)
    assert v.id == 42, f"Vertex id {v.id} doesn't match 42"
    assert v.x == pytest.approx(0.34)
    assert v.y == pytest.approx(0.58)
    assert v.z == pytest.approx(0.89)