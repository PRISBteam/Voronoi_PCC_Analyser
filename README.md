The code addressed the practical needs of creating discrete (combinatorial) cell complex (DCC) 
based on the Voronoi tessellation of space provided by the Neper software (https://neper.info). Such complexes arise from Voronoi tessellations of spatial domains around arbitrary sets of points, which ensure that each 1-cell is in the boundary of exactly three 2-cells and three 3-cells, and each 0-cell is in the boundary of exactly four 1-cells, six 2-cells and four 3-cells. This description is very close to real material microstructures and is widely used in molecular dynamics and other types of simulations.

## DCC definition and algebraic representation
In algebraic topology, a discrete topological n-complex is a collection of cells of dimensions k <= n, where every _k_-cell for any 0 < k <= n has a boundary formed by (_k_-1)-cells belonging to the complex. The co-boundary of every k-cell for 0 <= k < n is the set of (k+1)-cells whose boundaries contain the k-cell. In this terminology, 1-complex is a multigraph. Polyhedral complexes are a special class of regular quasi-convex discrete topological complexes, in the geometric realisation of which, 0-cells are identified with points or vertices, 1-cells with line segments or edges, 2-cells with planar polygons or faces, 3-cells with polyhedra or simply cells, etc. We restrict our consideration to the polyhedral 3-complexes whose 3-cells are convex polyhedra, whose 0-cells are in the boundary of exactly three 1-cells. 

The geometric properties of the DCC are encoded in the volumes of different cells: 1 for 0-cells, length for 1-cells, area for 2-cells, and volume for 3-cells. The topological properties of the DCC are encoded in the boundary operator B<sub>k</sub>, which maps all (_k_+1)-cells to the k-cells in their boundaries, taking into account cell orientations.
The algebraic realisation of the operator for the [_k_,(_k_+1)] pair of cells is referred to as the k-th incidence matrix, B<sub>k</sub>, which has N<sub>k</sub> rows (where N<sub>k</sub> denote the number of k-cells in a complex) and N<sub>k+1</sub> columns and contains 0, 1, -1, indicating non-adjacency, adjacency with agreeing and with opposite orientations, respectively, between k-cells and (k+1)-cells. A standard way is to decide on a consistent orientation of all top-dimensional cells, e.g., to select the positive orientation to be from interior to exterior of the 3-cells and assign arbitrary orientations for all lower-dimensional cells. There are exactly three options for the relation between _k_-cell and (_k_+1)-cell in an oriented complex: they are not coincident - encoded by 0; the k-cell is on the boundary (_k_+1)-cell, and they have consistent orientations, encoded by 1; the _k_-cell is on the boundary (_k_+1)-cell and they have opposite orientations, encoded by -1. The transpose of the k-th incidence matrix is a matrix representing the _k_-th co-boundary operator, which maps all _k_-cells to the (_k_+1)-cells in their co-boundaries.

## Terminal commands
The code can be launched by the usual terminal app on MAC, Windows or Linux. The first two commands create needed environment specified in the file _requirements.txt_. 

```
conda create --name neper-env --file requirements.txt
conda activate neper-env
python sparsemat.py --file filename.tess --dir my_dir
```

## Output files
The code generates a sparse representation of matrices: for any matrix element _a_(_i_, _j_) = _c_, the files of the matrices contain the list of triplets in the form (_i_, _j_, _c_). Indices start from 0, and, for instance, the line (5, 7, 1) in an adjacency matrix _A_k_ means that _k_-cell #6 is the neighbour of _k_-cell #8. For any incidence matrices B_k, a triplet (5, 7, 1) means that (_k_-1)-cell #6 is on the boundary of _k_-cell #8, and their orientations coincide (_c_ = -1 for the opposite orientations). 

### For the primal (Voronoi) complex:
All sparse matrices are stored in _*.txt_ files.

`A0.txt` - adjacency matrix for 0-cells (vertices)'A1.txt_ - adjacency matrix for 1-cells (edges)
`A2.txt` - adjacency matrix for 2-cells (faces)
`A3.txt` - adjacency matrix for 3-cells (polyhedra)

`B1.txt` - incidence matrix (0-cells are row indexes, 1-cells are columns)
`B2.txt` - incidence matrix (1-cells are row indexes, 2-cells are columns)
`B3.txt` - incidence matrix (2-cells are row indexes, 3-cells are columns)

In the 2D case, there are no _A3_ and _B3_ matrices.

`primal_Ncells.txt` - each row in the file corresponds to the numbers of different _k_-cells: the first row is the number of 0-cells,
second - is a number of 1-cells and so on.

`primal_normals.txt` - components of a normal vector for each face written in the format: ( face_id a b c ), where 
face_id coincide with the numeration of faces in _A2_ and _B2_ matrices; a, b and c - are the components of the normal vector of a face in a 3D cartesian coordinate system.

`primal_seeds.txt` - coordinates of the seed points of 3-cells used for Voronoi tessellation of space.

### For a dual (Delaunay) complex:

`a0.txt` - adjacency matrix for 0-cells (vertices)
`a1.txt` - adjacency matrix for 1-cells (edges) 
`a2.txt` - adjacency matrix for 2-cells (faces) 
`a3.txt` - adjacency matrix for 3-cells (polyhedra) 

`b1.txt` - incidence matrix (0-cells are row indexes, 1-cells are columns) 
`b2.txt` - incidence matrix (1-cells are row indexes, 2-cells are columns) 
`b3.txt` - incidence matrix (2-cells are row indexes, 3-cells are columns) 

In the 2D case, there are no _a3_ and _b3_ matrices.

`dual_Ncells.txt` - each row in the file corresponds to the numbers of different _k_-cells: the first row is the number of 0-cells,
second - is a number of 1-cells and so on.

`dual_normals.txt` - components of a normal vector for each face written in the format: ( face_id a b c ), where 
face_id coincide with the numeration of faces in _a2_ and _b2_ matrices; a, b and c - are the components of the normal vector of a face in a 3D cartesian coordinate system.

`dual_seeds.txt` - coordinates of the seed points of 3-cells used for Delaunay tessellation of space.


## Applications of DCCs
<ol>
<li> Kiprian Berbatov, Pieter D. Boom, Andrew L. Hazel, Andrey P. Jivkov, 2022. Diffusion in multi-dimensional solids using Forman’s combinatorial differential forms, Applied Mathematical Modelling, 110, 172-192. [doi: 10.1016/j.apm.2022.05.043.](https://doi.org/10.1016/j.apm.2022.05.043) </li>

<li> Pieter D. Boom, Odysseas Kosmas, Lee Margetts, Andrey P. Jivkov, 2022. A geometric formulation of linear elasticity based on discrete exterior calculus. International Journal of Solids and Structures, 236–237, 111345. [doi: 10.1016/j.ijsolstr.2021.111345.](https://doi.org/10.1016/j.ijsolstr.2021.111345) </li>

<li> Borodin, A.P. Jivkov, A.G. Sheinerman, M.Yu. Gutkin, 2021. Optimisation of rGO-enriched nanoceramics by combinatorial analysis. Materials & Design 212, 110191. [doi: 10.1016/j.matdes.2021.110191.](https://doi.org/10.1016/j.matdes.2021.110191) </li>

<li> S. Zhu, E.N. Borodin, A.P. Jivkov, 2021. Triple junctions network as the key structure for characterisation of SPD processed copper alloys. Materials & Design 198(24), 109352. [doi: 10.1016/j.matdes.2020.109352.](https://doi.org/10.1016/j.matdes.2020.109352) </li>

<li> D. Šeruga, O. Kosmas, A.P. Jivkov, 2020. Geometric modelling of elastic and elastic- plastic solids by separation of deformation energy and Prandtl operators, Int. J. Solids Struct. 198, 136–148. [doi: 10.1016/j.ijsolstr.2020.04.019.](https://doi.org/10.1016/j.ijsolstr.2020.04.019) </li>

<li> E. N. Borodin, A. P. Jivkov, 2019. Evolution of triple junctions’ network during severe plastic deformation of copper alloys – a discrete stochastic modelling. Philosophical Magazine 100(4), 467-485. [doi: 10.1080/14786435.2019.1695071.](https://doi.org/10.1080/14786435.2019.1695071) </li>

<li> I. Dassios, G. O’Keeffe, A.P. Jivkov, 2018. A mathematical model for elasticity using calculus on discrete manifolds, Math. Methods Appl. Sci. 41(18), 9057– 9070. [doi: 10.1002/mma.4892](https://doi.org/10.1002/mma.4892) </li>
</ol>

## Acknowledgements
This code has been created as a part of the EPSRC-funded project _EP/V022687/1 “Patterns recognition inside shear bands: tailoring microstructure against localisation” (PRISB)_.
