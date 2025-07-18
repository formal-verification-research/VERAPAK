#!/bin/bash

VERSION=$1
CATEGORY=$2
ONNX=$3
VNNLIB=$4
RESULTS_FILE=$5
TIMEOUT=$(echo "$6" | xargs)

if [ "$VERSION" != "v1" ]; then
  echo "Invalid version '$VERSION'. Expected 'v1'." >&2
  exit 1
fi

rm -f "$RESULTS_FILE"

ONNX_SHORT=${ONNX##*/}
ONNX_SHORT=${ONNX_SHORT%.*}
VNNLIB_SHORT=${VNNLIB##*/}
VNNLIB_SHORT=${VNNLIB_SHORT%.*}

echo ""
echo -n -e "\033[38;2;0;128;0m+++\033[0m"
echo -n " $ONNX_SHORT-$VNNLIB_SHORT "
echo -e "\033[38;2;0;128;0m+++\033[0m"
echo "Timeout: $TIMEOUT"

docker exec -it verapak python3 /src/VERAPAK --config_file /mnt/in/vnncomp.conf --output_dir=/mnt/out --timeout="$TIMEOUT" --halt_on_first

mkdir -p "verapak/out/$CATEGORY"
mv verapak/out/adversarial_examples.npy	"verapak/out/$CATEGORY/$ONNX_SHORT-$VNNLIB_SHORT.npy" 2> /dev/null
cp verapak/out/time_to_first.txt	"verapak/out/$CATEGORY/$ONNX_SHORT-$VNNLIB_SHORT.time"
cp verapak/out/report.csv		"verapak/out/$CATEGORY/$ONNX_SHORT-$VNNLIB_SHORT.csv"

REPORT=$(cat verapak/out/report.csv)

echo "$CATEGORY,$ONNX_SHORT,$VNNLIB_SHORT,$REPORT,verapak/out/$CATEGORY/$ONNX_SHORT-$VNNLIB_SHORT.npy" >> verapak/out/reports.csv

IFS=, read -ra arr <<< "$REPORT"
REPORT_ARRAY="${arr[0]};${arr[1]}"
REPORT_OUT=$(echo "$REPORT_ARRAY" | tr ";" "\n")

echo -n -e "\033[38;2;0;255;255mResult: ${arr[0]}\033[0m"
if [ -n "${arr[1]}" ]; then
	echo " (w/ witness)"
else
	echo ""
fi

#echo -n -e "\033[38;2;0;0;128m"
#echo -n "OUT: $REPORT_OUT"
#echo -e "\033[0m"
echo "$REPORT_OUT" > "$RESULTS_FILE"

echo -n -e "\033[38;2;0;128;0m---\033[0m"
echo -n " $ONNX_SHORT-$VNNLIB_SHORT "
echo -e "\033[38;2;0;128;0m---\033[0m"
echo ""


