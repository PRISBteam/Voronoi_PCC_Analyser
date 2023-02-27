#!/bin/bash

apt update && apt -y upgrade
apt -y install cmake libgsl-dev libomp-dev build-essential
# apt -y install povray gmsh asymptote

tar -xf v4.5.0.tar.gz
mkdir neper-4.5.0/src/build && cd neper-4.5.0/src/build
cmake .. && make && make install
