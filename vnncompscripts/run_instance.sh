#!/bin/bash

VERSION=$1
CATEGORY=$2
ONNX=$3
VNNLIB=$4
RESULTS_FILE=$5
TIMEOUT=`echo $6 | xargs`

rm -f $RESULTS_FILE

ONNX_SHORT=${ONNX##*/}
ONNX_SHORT=${ONNX_SHORT%.*}
VNNLIB_SHORT=${VNNLIB##*/}
VNNLIB_SHORT=${VNNLIB_SHORT%.*}

echo ""
echo -n -e "\033[38;2;0;128;0m+++\033[0m"
echo -n " $ONNX_SHORT-$VNNLIB_SHORT "
echo -e "\033[38;2;0;128;0m+++\033[0m"
echo "Timeout: $TIMEOUT"

docker run --rm --user $(id -u):$(id -g) -v $PWD:/src/in -v $PWD/out:/src/out verapak:latest python main.py --config_file=/src/in/vnncomp.conf --output_dir=/src/out --timeout=$TIMEOUT --halt_on_first

mkdir -p out/$CATEGORY
mv out/adversarial_examples.npy	out/$CATEGORY/$ONNX_SHORT-$VNNLIB_SHORT.npy 2> /dev/null
cp out/time_to_first.txt	out/$CATEGORY/$ONNX_SHORT-$VNNLIB_SHORT.time
cp out/report.csv		out/$CATEGORY/$ONNX_SHORT-$VNNLIB_SHORT.report

REPORT=`cat out/report.csv`

echo "$CATEGORY,$ONNX_SHORT,$VNNLIB_SHORT,$REPORT,out/$CATEGORY/$ONNX_SHORT-$VNNLIB_SHORT.npy" >> out/reports.csv

IFS=, read -a arr <<< "$REPORT"
REPORT_ARRAY="${arr[0]};${arr[1]}"
REPORT_OUT=`echo "$REPORT_ARRAY" | tr ";" "\n"`

echo -n -e "\033[38;2;0;255;255mResult: ${arr[0]}\033[0m"
if [ -n "${arr[1]}" ]; then
	echo " (w/ witness)"
else
	echo ""
fi

#echo -n -e "\033[38;2;0;0;128m"
#echo -n "OUT: $REPORT_OUT"
#echo -e "\033[0m"
echo $REPORT_OUT > $RESULTS_FILE

echo -n -e "\033[38;2;0;128;0m---\033[0m"
echo -n " $ONNX_SHORT-$VNNLIB_SHORT "
echo -e "\033[38;2;0;128;0m---\033[0m"
echo ""

