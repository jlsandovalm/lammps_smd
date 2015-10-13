#!/bin/sh

export OMPI_CXX=clang++-3.5

make package-update
make -j 4 $1
