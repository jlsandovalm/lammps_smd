#!/bin/sh

CC="/usr/bin/clang++-3.5"
if [ -e "$CC" ]
then
	export OMPI_CXX=$CC
else
	echo "$CC"
	echo "clang does not exist"
	return 1
fi

make package-update
make -j 4 $1
