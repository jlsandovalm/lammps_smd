#!/bin/sh

if [ -e "/usr/bin/clang++-3.6" ]
then
	export OMPI_CXX=/usr/bin/clang++-3.6
elif [ -e "/usr/bin/clang++-3.5" ]
then
	export OMPI_CXX=/usr/bin/clang++-3.5
fi

make package-update
make -j 4 $1
