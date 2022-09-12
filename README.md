```
conda create --name neper-env --file requirements.txt
conda activate neper-env
python sparsemat.py --file filename.tess --dir my_dir
```

Neper

Complexes

Output files

A0.txt - adjacency matrix for 0-cells (vertices)  
A1.txt - adjacency matrix for 1-cells (edges)  
A2.txt - adjacency matrix for 2-cells (faces)  
A3.txt - adjacency matrix for 3-cells (polyhedra)  

B1.txt - incidence matrix (0-cells are row indexes, 1-cells are columns)  
B2.txt - incidence matrix (1-cells are row indexes, 2-cells are columns)  
B3.txt - incidence matrix (2-cells are row indexes, 3-cells are columns)  

In 2D case there are no A3 and B3 matrices.

Matrices are stored in *.txt files.

Number of cells are stored in 'number_of_cells.txt' file.  
Each row corresponds to number of cells: first row is number of 0-cells,
second is number of 1-cells and so on.

Components of a normal vector for each face are stored in 'normals.txt'.  
Format: face_id a b c

seeds.txt

