#!/bin/bash

for thefile in *.py ; do
   grep -v "only remove this text" $thefile > $thefile.$$.tmp
   mv $thefile.$$.tmp $thefile
done
