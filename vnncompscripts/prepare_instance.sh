VERSION=$1
CATEGORY=$2
ONNX=$3
VNNLIB=$4

RGB_PRECISION="0.039215686274509803921568627451" # 1/255

case "$CATEGORY" in
	# VNNCOMP2021
	acasxu				) GRANULARITY="0.01x";; # 1/100th of each dimension's size
	cifar10_resnet			) GRANULARITY="$RGB_PRECISION";;
	cifar2020			) GRANULARITY="$RGB_PRECISION";;
#	eran				) GRANULARITY="";; # Uses default
	marabou-cifar10 		) GRANULARITY="$RGB_PRECISION";;
	mnistfc				) GRANULARITY="$RGB_PRECISION";;
#	nn4sys				) GRANULARITY="";; # Uses default
#	oval21				) GRANULARITY="";; # Uses default
#	test				) GRANULARITY="";; # Uses default
#	verivital			) GRANULARITY="";; # Uses default

	# VNNCOMP2022
	cifar100_tinyimagenet_resnet	) GRANULARITY="$RGB_PRECISION";;
	cifar_biasfield			) GRANULARITY="$RGB_PRECISION";;
	mnist_fc			) GRANULARITY="$RGB_PRECISION";;
#	oval21				) GRANULARITY="";; # See 2021 (uses default)
#	rl_benchmarks			) GRANULARITY="";; # Uses default
	sri_resnet_a			) GRANULARITY="$RGB_PRECISION";;
	sri_resnet_b			) GRANULARITY="$RGB_PRECISION";;

	# DEFAULT
	*				) GRANULARITY="0.001x";; # 1/1000th of each dimension's size
esac

mkdir -p verapak_docker
mkdir -p out

echo "graph :: /src/in/$ONNX" > vnncomp.conf
echo "vnnlib :: /src/in/$VNNLIB" >> vnncomp.conf
echo "abstraction_strategy :: rfgsm" >> vnncomp.conf
echo "verification_strategy :: eran" >> vnncomp.conf
#echo "granularity :: $GRANULARITY" >> vnncomp.conf # Note: When using ERAN, granularity isn't used

