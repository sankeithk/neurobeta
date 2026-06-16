#!/usr/bin/env

OUTPUT_DIR=$1

cd $OUTPUT_DIR

C1_FILE=$(dir | grep "^c1")
C2_FILE=$(dir | grep "^c2")
C3_FILE=$(dir | grep "^c3")

GM_VOL=$(fslstats ${C1_FILE} -M -V | awk '{print $1 * $2}')
WM_VOL=$(fslstats ${C2_FILE} -M -V | awk '{print $1 * $2}')
CSF_VOL=$(fslstats ${C3_FILE} -M -V | awk '{print $1 * $2}')

# 2. Sum them to get the ICV
ICV=$(awk "BEGIN {print ${GM_VOL} + ${WM_VOL} + ${CSF_VOL}}")
echo "Estimated intracranial volume (in mm^3) = ${ICV}"
echo ${ICV} > "${OUTPUT_DIR}/icv.txt"