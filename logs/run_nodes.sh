#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
echo "nodes begin $(date +%H:%M:%S)" >> progress.txt
( time python3 fit_joint.py nodes ) > nodes.log 2>&1
echo "nodes done $(date +%H:%M:%S) exit=$?" >> progress.txt
