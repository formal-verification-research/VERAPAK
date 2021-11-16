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

mkdir -p verapak_docker
mkdir -p out

echo "graph :: /src/in/$ONNX" > vnncomp.conf
echo "vnnlib :: /src/in/$VNNLIB" >> vnncomp.conf
echo "abstraction_strategy :: rfgsm" >> vnncomp.conf
echo "verification_strategy :: discrete_search" >> vnncomp.conf
echo "granularity :: $GRANULARITY" >> vnncomp.conf

