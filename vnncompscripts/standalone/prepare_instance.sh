#!/bin/bash

VERSION=$1
CATEGORY=$2
ONNX=$3
VNNLIB=$4

if [ "$VERSION" != "v1" ]; then
  echo "Invalid version '$VERSION'. Expected 'v1'." >&2
  exit 1
fi

#case "$CATEGORY" in
#  # VNNCOMP2021
#  acasxu          ) GRANULARITY="0.01x";; # 1/100th of each dimension's size
#  cifar10_resnet  ) GRANULARITY="$RGB_PRECISION";;
#  cifar2020       ) GRANULARITY="$RGB_PRECISION";;
##  eran            ) GRANULARITY="";; # Uses default
#  marabou-cifar10 ) GRANULARITY="$RGB_PRECISION";;
#  mnistfc         ) GRANULARITY="$RGB_PRECISION";;
##  nn4sys          ) GRANULARITY="";; # Uses default
##  oval21          ) GRANULARITY="";; # Uses default
##  test            ) GRANULARITY="";; # Uses default
##  verivital       ) GRANULARITY="";; # Uses default
#
#  # VNNCOMP2022
#  cifar100_tinyimagenet_resnet	) GRANULARITY="$RGB_PRECISION";;
#  cifar_biasfield ) GRANULARITY="$RGB_PRECISION";;
#  mnist_fc        ) GRANULARITY="$RGB_PRECISION";;
##  oval21          ) GRANULARITY="";; # See 2021 (uses default)
##  rl_benchmarks   ) GRANULARITY="";; # Uses default
#  sri_resnet_a    ) GRANULARITY="$RGB_PRECISION";;
#  sri_resnet_b    ) GRANULARITY="$RGB_PRECISION";;
#
#  # DEFAULT
#  *               ) GRANULARITY="0.001x";; # 1/1000th of each dimension's size
#esac

ONNX_SHORT=${ONNX##*/}
VNNLIB_SHORT=${VNNLIB##*/}

cp "$ONNX" "verapak/in/$ONNX_SHORT"
cp "$VNNLIB" "verapak/in/$VNNLIB_SHORT"

{
  echo "graph :: /mnt/in/$ONNX_SHORT";
  echo "vnnlib :: /mnt/in/$VNNLIB_SHORT";
  echo "abstraction_strategy :: rfgsm";
  echo "verification_strategy :: eran";
# echo "granularity :: $GRANULARITY";
} > verapak/in/vnncomp.conf