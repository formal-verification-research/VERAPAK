VERSION=$1
CATEGORY=$2
ONNX=$3
VNNLIB=$4

case "$CATEGORY" in
	acasxu		) GRANULARITY="0.01x";;
	cifar10_resnet	) GRANULARITY="0.00390625";;
	cifar2020	) GRANULARITY="0.00390625";;
	eran		) GRANULARITY="0.001x";;
	marabou-cifar10 ) GRANULARITY="0.00390625";;
	mnistfc		) GRANULARITY="0.00390625";;
	nn4sys		) GRANULARITY="0.001x";;
	oval21		) GRANULARITY="0.001x";;
	test		) GRANULARITY="0.001x";;
	verivital	) GRANULARITY="0.001x";;
	*		) GRANULARITY="0.001x";;
esac

echo -e "graph :: $ONNX\nvnnlib :: $VNNLIB\nabstraction_strategy :: rfgsm\nverification_strategy :: discrete_search\ngranularity :: $GRANULARITY" > vnncomptests/vnncomp.conf
