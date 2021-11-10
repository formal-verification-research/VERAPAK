VERSION=$1
CATEGORY=$2
ONNX=$3
VNNLIB=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo $TIMEOUT

docker run --rm -it -v $PWD/vnncomptests:/src/vnncomptests -v $PWD/out:/src/out verapak:latest python main.py --config_file=vnncomptests/vnncomp.conf --output_dir=out --timeout=$TIMEOUT

mkdir out/$CATEGORY 2> /dev/null
mv out/adversarial_example.npy	out/$CATEGORY/$ONNX-$VNNLIB.npy
cp out/time_to_first.txt	out/$CATEGORY/$ONNX-$VNNLIB.time
cp out/report.txt		out/$CATEGORY/$ONNX-$VNNLIB.report
mv out/time_to_first.txt	$RESULTS_FILE.time
mv out/report.txt		$RESULTS_FILE

