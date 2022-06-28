import matplotlib.pyplot as plt
import pytest

from matgen import core

def test_create_ax():
    """
    """
    ax = core._create_ax()
    plt.savefig('tests/figures/new-ax-default.png')

    ax = core._create_ax(dim=3)
    plt.savefig('tests/figures/new-ax-3D.png')

    ax = core._create_ax(dim=2, figsize=(16,16))
    plt.savefig('tests/figures/new-ax-2D-size-16.png')

    ax = core._create_ax(dim=3, figsize=(4,4))
    plt.savefig('tests/figures/new-ax-3D-size-4.png')


def test_Vertex():
    """
    """
    v = core.Vertex(id=42, x=0.34, y=0.58, z=0.89)
    assert v.id == 42, f"Vertex id {v.id} doesn't match 42"
    assert v.x == pytest.approx(0.34)
    assert v.y == pytest.approx(0.58)
    assert v.z == pytest.approx(0.89)
    assert v.coord == pytest.approx((0.34, 0.58, 0.89))
    assert v.coord2D == pytest.approx((0.34, 0.58))
    assert v.__str__() == "Vertex(id=42)"

    v.add_neighbor(21)
    v.add_neighbor(34)
    v.add_neighbors([1, 2])
    v.add_neighbors([5, 1, 3])
    v.add_neighbor(2)
    assert len(v.neighbor_ids) == 6
    for v_id in [1, 2, 3, 5, 21, 34]:
        assert v_id in v.neighbor_ids
    
    v.add_incident_edge(21)
    v.add_incident_edge(34)
    v.add_incident_edges([1, 2])
    v.add_incident_edges([5, 1, 3])
    v.add_incident_edge(5)
    assert len(v.e_ids) == 6
    assert v.get_degree() == 6
    assert len(v.incident_cells) == 6
    for e_id in [1, 2, 3, 5, 21, 34]:
        assert e_id in v.e_ids
    for e_id in [1, 2, 3, 5, 21, 34]:
        assert e_id in v.incident_cells

    assert not v.is_external
    v.set_external(True)
    assert v.is_external
    v.set_external(False)
    assert not v.is_external
    
    v.set_junction_type('E')
    assert v.junction_type == 'E'
    v.set_junction_type('J1')
    assert v.junction_type == 'J1'

    points = [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.0, 1.0),
        (0.5, 0.0),
        (1.0, 0.0)
    ]
    ax = core.Vertex(id=21, x=0.5, y=0.5).plot(dim=2, color='k', label=21)
    i = 0
    for x, y in points:
        i += 1
        v = core.Vertex(id=i, x=x, y=y)
        ax = v.plot(dim=2, ax=ax, label=i)
    ax.legend(loc='best')
    plt.savefig('tests/figures/vertices-2D-6.png')

    points = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0)
    ]
    ax = core.Vertex(id=21, x=0.5, y=0.5, z=0.5).plot(
        dim=3, color='k', label=21
    )
    i = 0
    for x, y, z in points:
        i += 1
        v = core.Vertex(id=i, x=x, y=y, z=z)
        ax = v.plot(dim=3, ax=ax, label=i)
    ax.legend(loc='best')
    plt.savefig('tests/figures/vertices-3D-9.png')

    ax = core.Vertex(id=21, x=0.5, y=0.5, z=0.5).plot(
        dim=2, color='r', label=22
    )
    ax.legend(loc='best')
    plt.savefig('tests/figures/vertices-2D-1.png')

    filename = 'tests/test_data/n8-id1.tess'
    _vertices = core.Vertex.from_tess_file(filename)
    assert len(_vertices.keys()) == 41
    assert _vertices[20].z == pytest.approx(0.581244510538)
    assert _vertices[33].y == pytest.approx(0.377184567183)
    assert _vertices[39].x == pytest.approx(0.658946058940)
    



def test_CellComplex():
    """
    """
    filename = 'tests/test_data/n8-id1.tess'
    c = core.CellComplex(filename)

    assert c.source_file == filename, "Filename doesn't match"