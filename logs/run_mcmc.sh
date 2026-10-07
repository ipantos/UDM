#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
for seg in 1 2 3 4 5 6 7 8; do
  echo "mcmc segment $seg begin $(date +%H:%M:%S)" >> progress.txt
  ( time python3 fit_joint.py mcmc 5000 ) > mcmc_seg$seg.log 2>&1
  rc=$?
  echo "mcmc segment $seg done $(date +%H:%M:%S) exit=$rc" >> progress.txt
  if [ $rc -ne 0 ]; then echo "FAILED segment $seg" >> progress.txt; exit 1; fi
  cp -r runs runs_backup_seg$seg
done
echo "mcmc summary begin $(date +%H:%M:%S)" >> progress.txt
( time python3 fit_joint.py summary ) > summary.log 2>&1
echo "summary done $(date +%H:%M:%S) exit=$?" >> progress.txt
( time python3 fit_joint.py validate ) > validate.log 2>&1
echo "validate done $(date +%H:%M:%S) exit=$?" >> progress.txt
echo "ALL DONE $(date +%H:%M:%S)" >> progress.txt
