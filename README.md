# Polyhedral Cell Complex (PCC) Generator tool

<p> The code addressed the practical needs of creating discrete (combinatorial) cell complexes (DCCs) 
based on the both regular (cubic or octahedron) and Laguerre-Voronoi tessellations of space provided by the <a href="https://neper.info" target=”_blank”> Neper </a> software. In particular, Voronoi tessellations with the corresponding DCCs provide a very close representation of the real material microstructures and are widely used in molecular dynamics and other types of simulations. Such complexes arise from the tessellations of spatial domains around arbitrary sets of points, which ensure that each 1-cell is in the boundary of exactly three 2-cells and three 3-cells, and each 0-cell is in the boundary of exactly four 1-cells, six 2-cells and four 3-cells. </p>

## 1. DCC definition and algebraic representation
An excellent simple introduction to the area of DCCs with their various applications is given in the <a href="https://link.springer.com/book/10.1007/978-1-84996-290-2" target="_blank"> book </a> of Leo Grady and Jonathan Polimeni _“Discrete Calculus. Applied Analysis on Graphs for Computational Science. (2010)_ Below are just a few notes necessary for understanding the output of the code.

<ul>
<li> In algebraic topology, a discrete topological n-complex is a collection of cells of dimensions <i> k &leq; n </i>, where every k-cell for any <i> 0 < k &leq; n </i> has a boundary formed by (k-1)-cells belonging to the complex. The co-boundary of every k-cell for <i> 0 &leq; k < n </i> is the set of (k+1)-cells whose boundaries contain the k-cell. In this terminology, 1-complex is a graph. Polyhedral complexes are a special class of regular quasi-convex discrete topological complexes, in the geometric realisation of which 0-cells are identified with points or vertices, 1-cells with line segments or edges, 2-cells with planar polygons or faces, 3-cells with polyhedra or simply cells, etc. We restrict our consideration to the polyhedral 3-complexes whose 3-cells are convex polyhedra with 2-cells in the boundary of exactly two 3-cells. An assembly of polyhedrons is a geometric realisation of a combinatorial structure referred to as a cell complex in algebraic topology. </li>
  
<li> The geometric properties of the DCC are encoded in the volumes of different cells: 1 for 0-cells, length for 1-cells, area for 2-cells, and volume for 3-cells. The topological properties of the DCC are encoded in the boundary operator B<sub>k</sub>, which maps all (k+1)-cells to the k-cells in their boundaries, taking into account cell orientations. The algebraic realisation of the operator for the [k,(k+1)] pair of cells is referred to as the k-th incidence matrix, B<sub>k</sub>, which has N<sub>k</sub> rows (where N<sub>k</sub> denote the number of k-cells in a complex) and N<sub>k+1</sub> columns and contains 0, 1, -1, indicating non-adjacency, adjacency with agreeing and with opposite orientations, respectively, between k-cells and (k+1)-cells. The transpose of the k-th incidence matrix, b<sub>k</sub> =  B<sub>k</sub><sup>T</sup>, is a matrix representing the k-th co-boundary operator, which maps all k-cells to the (k+1)-cells in their co-boundaries. </li>

<li> A standard way is to decide on a consistent orientation of all top-dimensional cells, e.g., to select the positive orientation to be from interior to exterior of the 3-cells and assign arbitrary orientations for all lower-dimensional cells. There are exactly three options for the relation between k-cell and (k+1)-cell in an oriented complex: they are not coincident - encoded by 0; the k-cell is on the boundary (k+1)-cell, and they have consistent orientations, encoded by 1; the k-cell is on the boundary (k+1)-cell and they have opposite orientations, encoded by -1. The transpose of the k-th incidence matrix is a matrix representing the k-th co-boundary operator, which maps all k-cells to the (k+1)-cells in their co-boundaries. </li>

<li> The k-th combinatorial <i>Laplacian</i> (Laplace–de Rham operator) can be written as <br>
<i> L<sub>k</sub> = b<sub>k-1</sub> B<sub>k-1</sub> + B<sub>k</sub> b<sub>k</sub>  </i> <br>
and it maps all k-cells to themselves, collecting local connectivity information. One important application of the combinatorial Laplacians is in calculating <a href="https://link.springer.com/article/10.1007/s00454-002-0743-x" target=”_blank”> combinatorial curvatures </a>. Since the Laplacians are symmetric positive semi-definite matrices, their eigenvalues are real. The spectra of eigenvalues can be used to classify discrete topologies, with two topologies considered equivalent when they have the same Laplacians’ spectra. </li>
</ul>

## 2. Terminal commands
The usual terminal app can launch the code on MAC, Windows or Linux. The first two commands create the needed environment specified in the file `requirements.txt`

```
conda create --name neper-env --file requirements.txt
conda activate neper-env
python sparsemat.py --file <filename.tess> --dir <my_dir>
```
Here <filename.tess> is the full path (including the file name) to the <i>*.tess</i> file generated by Neper software, and <my_dir> is the full path to the chosen output directory. Finally, all output files will be written to the <my_dir> output directory.

The required Conda packages can be downloaded and installed directly from the Anaconda and Minicoda projects <a href="https://conda.io/projects/conda/en/latest/user-guide/install/download.html" target="_blank"> webpages</a>.
  
The code has been tested for the Neper output <i>*.tess</i> files generated by Neper's versions 4.3.1 and 4.5.0.

## 3. Output files
The code generates a sparse representation of matrices: for any matrix element _a_(_i_, _j_) = _c_, the files of the matrices contain the list of triplets in the form (_i_, _j_, _c_). The enumeration of indices starts from 0, and, for instance, the line "5, 7, 1" in the adjacency matrix A<sub>k</sub> means that the _k_-cell #6 is the neighbour of the _k_-cell #8. For any incidence matrices B<sub>k</sub>,  the same triplet "5, 7, 1" means that the (_k_-1)-cell #6 is on the boundary of the _k_-cell #8, and their orientations coincide (_c_ = -1 for the opposite orientations). 
  
The Voronoi tesselation provided by Neper supposed to be a <i>dual</i> complex in terms of algebraic topology and so all the other tessellations provided by the Neper output with the <a href="https://neper.info/doc/neper_t.html#morphology-options" target="_blank"> morphology </a> option <i> -morpho <morphology> </i> like <i> cube, square, tocta, lamellar, etc. </i> different from <i>voronoi</i>.

### For the dual (voronoi, cube, tocta, etc.) complex:
All sparse matrices are stored in <i>*.txt</i> files.

`A0.txt` - 0-adjacency matrix for 0-cells (vertices) <br>
`A1.txt` - 1-adjacency matrix for 1-cells (edges) <br>
`A2.txt` - 2-adjacency matrix for 2-cells (faces) <br>
`A3.txt` - 3-adjacency matrix for 3-cells (polyhedra) <br>

`B1.txt` - incidence matrix or boundary operator (0-cells are row indexes, 1-cells are columns) <br>
`B2.txt` - incidence matrix or boundary operator (1-cells are row indexes, 2-cells are columns) <br>
`B3.txt` - incidence matrix or boundary operator (2-cells are row indexes, 3-cells are columns) <br>

In the 2D case, there are no _A3_ and _B3_ matrices.

`voro_Ncells.txt` - each row in the file corresponds to the numbers of different k-cells: the first row is the number of 0-cells,
second - is a number of 1-cells and so on.

`voro_normals.txt` - components of a normal vector for each face written in the format: <i> (face_id a b c) </i>, where 
<i>face_id</i> coincide with the numeration of faces in _A2_ and _B2_ matrices; _a_, _b_ and _c_ - are the components of the normal vector of a face in a 3D cartesian coordinate system.

`voro_seeds.txt` - coordinates of the seed points of 3-cells used for Voronoi tessellation of space.

### For a primal (delaunay triangle, cubic, etc.) complex:

`AC0.txt` - adjacency matrix for 0-cells (vertices) <br>
`AC1.txt` - adjacency matrix for 1-cells (edges) <br>
`AC2.txt` - adjacency matrix for 2-cells (faces) <br>
`AC3.txt` - adjacency matrix for 3-cells (polyhedra) <br>

`BC1.txt` - incidence matrix or co-boundary operator (0-cells are row indexes, 1-cells are columns) <br>
`BC2.txt` - incidence matrix or co-boundary operator (1-cells are row indexes, 2-cells are columns) <br>
`BC3.txt` - incidence matrix or co-boundary operator (2-cells are row indexes, 3-cells are columns) <br>

In the 2D case, there are no _AC3_ and _BC3_ matrices.

`delau_Ncells.txt` - each row in the file corresponds to the numbers of different k-cells: the first row is the number of 0-cells,
second - is a number of 1-cells and so on.

<!-- `delone_normals.txt` - components of a normal vector for each face written in the format: <i> (face_id a b c) </i>, where 
face_id coincide with the numeration of faces in _a2_ and _b2_ matrices; _a_, _b_ and _c_ - are the components of the normal vector of a face in a 3D cartesian coordinate system. -->

`delau_seeds.txt` - coordinates of the seed points of 3-cells used for Delaunay tessellation of space.

### For the whole complex:
`L0.txt` - 0-Laplacian matrix with the dimension of 0-cells (vertices) <br>
`L1.txt` - 1-Laplacian matrix with the dimension of 1-cells (edges) <br>
`L2.txt` - 2-Laplacian matrix with the dimension of 2-cells (faces) <br>

## 4. Tips and tricks
<ul>
<li> The metric information like the volumes of all 3-cells and areas of all 2-cells can be obtain directly from the Neper output using  <a href="https://neper.info/doc/neper_t.html#cmdoption-statcell" target="_blank"> statcell </a> and statface options with the corresponding <a href="https://neper.info/doc/exprskeys.html#tessellation-keys" target="_blank"> keys </a> like "-statcell vol -statface area" or  providing the corresponding values for every k-cell in the complex. In this case, the terminal command may look like 
  
```
   neper -T -n 300 -id 1 -dim 3 -statcell vol -statface area
```
Please, see more <a href="https://neper.info/doc/neper_t.html#examples" target="_blank"> examples </a> on the Neper webpage.
</li>
  
<li> Using the file ``seeds.txt`` with some specific set of seed points a new Neper tessellation can be performed. The terminal command creating a complex with coordinates of the seed points as the centres of 3-cells may looks like 
 
```
neper -T -n <number of seeds> -id 1 -statcell vol -statface area -domain "cube(1.0,1.0,1.0)" -morphooptiini "coo:file(seeds.txt)"
```
You must call Neper from the folder (cd <path to the directory containing the file "seeds.txt">) containing the ``seeds.txt`` file, or write the whole path instead of the file name in the <i>coo:file(<path to seeds.txt>)</i> command.
</li>

<li> More flexibility in the tesselation provide the <a href="https://neper.info/doc/neper_t.html#examples" target="_blank"> transformation </a> options of the Neper. In particular, for the creation of a 2D complex as a plane cut of the 3D one, the <i> slice(d,a,b,c)</i> function can be used as it is shown below for the half-cut of the Voronoi complex containing 1000 grains:

```
neper -T -n 1000 -id 1 -domain "cube(1.0,1.0,1.0)" -transform "slice(0.5,0,0,1)" -dim 3 -statcell area; \
neper -V n1000-id1.tess -datacelltrs 0.5  -print DCC_slice
```
Here <i> d, a, b </i>, and <i>c</i> are parameters in the corresponding equation of a plane <i> ax + by + cz = d</i> and it is worth to be mentioned here that the normal vector of this plane is <i>n = (a,b,c)</i>.
</li>
  
</ul>

## 5. Examples
The folder with several examples contains discrete cell complexes already created by Neper and processed with DCC Generator Tool. The command below shows three terminal commands launching the creation of the Vorinoi dual complex containing 5000 3-cells and its further processing using DCC Generator Tool:

```
neper -T -n 5000 -dim 3 -id 1 -ori uniform -statcell vol -statface area; \
neper -V n5000-id1.tess -datacelltrs 0.5  -print DCC_voronoi_5000; \
python <path/to/the/directory>/Voronoi_DCC_Analyser/sparsemat.py --file <path/to/the/directory>/n5000-id1.tess --dir <path/to/right/directory>;
```
All the amounts of k-cells in the complex can be taken directly from the `voro_Ncells.txt` or `delau_Ncells.txt` files.

## Applications of DCCs
<ol>
<li> K. Berbatov, P.D. Boom, A.L. Hazel, A.P. Jivkov, 2022. Diffusion in multi-dimensional solids using Forman’s combinatorial differential forms. Applied Mathematical Modelling 110, 172-192. [doi: 10.1016/j.apm.2022.05.043.](https://doi.org/10.1016/j.apm.2022.05.043) </li>

<li> P.D. Boom, O. Kosmas, L. Margetts, A.P. Jivkov, 2022. A geometric formulation of linear elasticity based on discrete exterior calculus. International Journal of Solids and Structures 236–237, 111345. [doi: 10.1016/j.ijsolstr.2021.111345.](https://doi.org/10.1016/j.ijsolstr.2021.111345) </li>

<li> E.N. Borodin, A.P. Jivkov, A.G. Sheinerman, M.Yu. Gutkin, 2021. Optimisation of rGO-enriched nanoceramics by combinatorial analysis. Materials & Design 212, 110191. [doi: 10.1016/j.matdes.2021.110191.](https://doi.org/10.1016/j.matdes.2021.110191) </li>
 
<li> S. Zhu, E.N. Borodin, A.P. Jivkov, 2021. Triple junctions network as the key structure for characterisation of SPD processed copper alloys. Materials & Design 198(24), 109352. [doi: 10.1016/j.matdes.2020.109352.](https://doi.org/10.1016/j.matdes.2020.109352) </li>

<li> D. Šeruga, O. Kosmas, A.P. Jivkov, 2020. Geometric modelling of elastic and elastic-plastic solids by separation of deformation energy and Prandtl operators. International Journal of Solids and Structures 198, 136–148. [doi: 10.1016/j.ijsolstr.2020.04.019.](https://doi.org/10.1016/j.ijsolstr.2020.04.019) </li>

<li> E. N. Borodin, A. P. Jivkov, 2019. Evolution of triple junctions’ network during severe plastic deformation of copper alloys – a discrete stochastic modelling. Philosophical Magazine 100(4), 467-485. </li> [doi: 10.1080/14786435.2019.1695071.](https://doi.org/10.1080/14786435.2019.1695071) 

<li> I. Dassios, G. O’Keeffe, A.P. Jivkov, 2018. A mathematical model for elasticity using calculus on discrete manifolds. Math. Methods Appl. Sci. 41(18), 9057– 9070. </li> [doi: 10.1002/mma.4892](https://doi.org/10.1002/mma.4892) 
</ol>

## Acknowledgements
This code has been created as a part of the EPSRC funded projects <a href="https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/V022687/1" target="_blank"> EP/V022687/1 </a> <i>“Patterns recognition inside shear bands: tailoring microstructure against localisation”</i> (PRISB) and <a href="https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/N026136/1" target="_blank"> EP/N026136/1 </a> <i>"Geometric Mechanics of Solids: new analysis of modern engineering materials"</i> (GEMS).

## License
Distributed under the GNU General Public License v3.0. See `LICENSE.txt` for more information.
  
## Contacts
<a href="mailto: prisb.team@gmail.com" target="_blank"> Send e-mail</a> <br>
Dr Oleg Bushuev (technical support) <br>
Dr Elijah Borodin (any other questions) <br>
