#!/bin/bash
echo "Enter n"
read n
echo "Enter id"
read id
echo "Enter dim"
read dim
neper -T -n $n -id $id -dim $dim
neper -V n$n-id$id.tess -print img$n-$id
# conda activate neper-env
python3 ~/GitHub/Voronoi_DCC_Analyser/matgen/sparsemat.py n$n-id$id.tess . -o

