#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=projects/sketchto3D/sketch_solver.prototxt
