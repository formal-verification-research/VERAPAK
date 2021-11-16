VERSION=$1
CATEGORY=$2
ONNX=$3
VNNLIB=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo $TIMEOUT
echo ${ONNX##*/}-${VNNLIB##*/}

docker run --rm --user $(id -u):$(id -g) -v $PWD:/src/in -v $PWD/out:/src/out verapak:latest python main.py --config_file=/src/in/vnncomp.conf --output_dir=/src/out --timeout=$TIMEOUT --halt_on_first || exit 2

mkdir -p out/$CATEGORY
mv out/adversarial_examples.npy	out/$CATEGORY/${ONNX##*/}-${VNNLIB##*/}.npy
cp out/time_to_first.txt	out/$CATEGORY/${ONNX##*/}-${VNNLIB##*/}.time
cp out/report.txt		out/$CATEGORY/${ONNX##*/}-${VNNLIB##*/}.report
mv -f out/time_to_first.txt	$RESULTS_FILE.time
mv -f out/report.txt		$RESULTS_FILE

