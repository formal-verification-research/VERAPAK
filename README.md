# VeRAPAk <img src=https://formal-verification-research.github.io/assets/logo/logo.png width=40 style="float: right;">
### **V**erification for **R**obust neural networks through **A**bstraction, **P**artitioning, and **A**ttack methods

VeRAPAk is an algorithmic framework for optimizing formal verification techniques for deep neural networks when it is used for verifying the local adversarial robustness property of classification problems. 


[Installation](#installation)\
[Usage](#usage) \
[Example](#example)\
[Abstraction, Verification, and Partitioning Techniques](#abstraction--verification-and-partitioning-techniques) \
[Building from source](#Building-from-source) \
[Contact](#contact) 



# Installation
Clone the VeRAPAk repository via git as follows:
```
git clone https://github.com/formal-verification-research/VERAPAK.git
cd VERAPAK
```
Build VeRAPAk docker image and run container
```
docker build -t <image_name> .
docker run --network host -it <image_name>
```

Note: VeRAPAk can be used in the CLI directly with ```VERAPAK```command when running within Docker

Image can also be found on Docker Hub
```
docker pull yodarocks1/verapak:latest
```

# Usage


```--config_file```: Allows passing of arguments in a .conf file using ```key::value``` pairs one-per-line

```--vnnlib VNNLIB```: VNNLIB region definition.  Parse out the centerpoint, intended
label, and radii, and use them if no others are
provided in the config_file or command line flags.

```--output_dir```: Path to output directory where adversarial examples
will be stored

```--initial_point```: Point around which robustness verification will occur (example of what a point is?)

```--label LABEL```: Intended class label number (use index of logit). If
neither this nor a constraint_file are provided, class
of initial_point is assumed as the intended class.

```--constraint_file```: Output constraints in a file one-per-line. They should
take the form <label> [label...] <constraint> [other],
where `label` is a 0-based output index, `constraint`
is one of '>', '<', 'min', 'max', 'notmin', 'notmax',
and '<=', and `other` is a floating point number
(instead of a label) when it ends in `f` or contains a
decimal point (`.`). #TODO

```--graph```: Path to serialized DNN model (ONNX, TF, etc) (etc? what formats are supported?) #TODO

```--radius```: Radius (single value or per dimension array) around
initial point where robustness verification will occur

```--num_abstractions```: Number of abstraction points to generate each pass

```--verification_strategy```: [Verification strategy](#verification)  (`Eran` OR `Discrete search`)
<details> <summary> Discrete Search Parameters </summary>
<hr>

```--verification_point_threshold``` Threshold number of discrete points under which
verification should occur

```--granularity``` Granularity (single value or per dimension array): a
valid discretization of the input space (8 bit image
-> 1/255)
<hr>
 </details>

<details> <summary> ERAN Parameters </summary> 
<hr>

```--eran_timeout``` ERAN timeout: 0 for no timeout, negative timeout is a multiple of the full timeout
<hr>
</details>

\
```--abstraction_strategy```: [Abstraction strategy](#abstractions)  (`center`, `fgsm`,
`random`, `rfgsm`)

<details> <summary> FGSM Parameters </summary>
<hr>

```--granularity``` Granularity (single value or per dimension array): a
valid discretization of the input space (8 bit image
-> 1/255)
<hr>

</details>

<details> <summary> R-FGSM Parameters </summary>
<hr>

```--granularity``` Granularity (single value or per dimension array): a
valid discretization of the input space (8 bit image
-> 1/255)

```--balance_factor``` Balance factor (1.0 -> all FGSM, 0.0 -> purely random)

```--dimension_ranking_strategy``` Dimension ranking strategy: (
`gradient_based`, `by_index`, `largest_first`)
<hr>
</details>

\
```--partitioning_strategy``` [Partitioning strategy](#partitioning) (`largest_first`)

```--partitioning_divisor``` Number of divisions on each dimension during
partitioning

```--partitioning_dimensions``` Number of dimensions to partition

`--region_lower_bound` Lower bound for the region

`--region_upper_bound` Upper bound for the region

`--timeout TIMEOUT` Number of seconds to run the program before reporting all found adversarial examples and timing out (set to
0 for 'run until interrupted')

`--report_interval_seconds` Number of seconds between status reports

`--halt_on_first` [{loose,strict,none}]
If given, halt on the first adversarial example (`loose`, `strict`, `none`)

`--no-color`  Remove colorization of the output (currently only
implemented for errors)


Command line arguments can be passed directly:
~~~
verapak --graph nets/net2bverified.onnx --vnnlib cifar10_spec_idx_1_eps_0.00784.vnnlib --abstraction_strategy rfgsm --verification_strategy eran --output_dir "./src/out"

~~~

OR

Arguments can also be passed in a .conf file using ```keyword :: value``` for pairs one per line:
```sh
verapak --config_file=test_a.conf 
```


a.conf:
```
graph :: vnncomp/cifar2020/nets/convBigRELU__PGD.onnx
vnnlib :: vnncomp/cifar2020/specs/cifar10_spec_idx_1_eps_0.00784.vnnlib
abstraction_strategy :: rfgsm
verification_strategy :: eran
```



# Example
A handful of example networks and strategies can be found in the `examples` directory \
Below we show how to run VeRAPAk using these example networks and strategies:

```sh
cd ./examples
verapak --config_file=test_a.conf --output_dir="/src/out" --timeout 0 --halt_on_first loose
```
Resulting output:
```
Final Report
#############################
@ 105.7587821483612
Percent unknown: 0.0%
Percent known safe: 0.0%
Percent known unsafe: 0.0%
Percent partially unsafe: 100.0%
Adversarial examples: 0
```

# Abstraction,  Verification, and Partitioning Techniques

### Abstractions
The abstraction strategy is a swappable module that alters the method of choosing points that are most likely refute the adversarial robustness property by generating an adversarial example.
<details>
<summary>Center Point</summary>
 
 * This strategy assumes the central point is the most representative of the given region. May perform well under the assumption that adversarial examples commonly exist in clusters

</details>

<details>
<summary>Fast Gradient Sign Method (FGSM)</summary>

* White box attack requiring access to the network to create an adversarial example

* Utilizes the gradients of the network with given input to create new image in which loss is maximized

$` adv\_x = x + \epsilon * sign(\partial x J(\theta,x,y))`$

Where, 
* adv\_x: Adversarial Image
* x: Original image
* y: Original input label
* $\epsilon$: Multiplier to ensure the perturbations are small
* $\theta$: Model Parameters
* J: Loss

I. Goodfellow, J. Shlens, and C. Szegedy, “Published as a conference paper at ICLR 2015 EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES,” 2015. Available: https://arxiv.org/pdf/1412.6572
‌

</details>

<details>
<summary>Random Point</summary>

* Naive approach to abstraction in which points are selected at random. Utilized as a control for experiments rather than viable abstraction method.
</details>

<details>
<summary>Random Fast Gradient Sign Method (R-FGSM)</summary>

* FGSM strategy suffers from minimal varience in repeated calls with an identical base point, limiting the potential use for adversial generation
* Our proposed solution introduces dimension selection heuristics to allow randomized selection providing increased varience and increase success rate of adversial generation


</details>


### Verification

<details>
<summary>ETH Robustness Analyzer for Nueral Networks (ERAN)</summary>

* ERAN is a state-of-the-art analyzer for verifying neural networks making use of abstract interpretation to analyze networks with feedforward, convolutional, and residual layers against input perturbations. 
* ERAN supports multiple analysis techniques including: DeepZ, DeepPoly, GPUPoly, RefineZono, and RefinePoly/RefineGPUPoly.

* ERAN source: https://github.com/eth-sri/eran \
\
G. Singh, T. Gehr, M. Püschel, and M. Vechev, ‘An abstract domain for certifying neural networks’, Proc. ACM Program. Lang., vol. 3, no. POPL, Jan. 2019.

</details>



<details>
<summary>Discrete search</summary>
</details>

### Partitioning
Regions are partitioned by the partitioning engine which determines the number of resulting subregions
<details>
<summary>Largest First</summary>

* Largest first partitioning strategy subdivides the largest regions first into equal subregions i.e. 3 largest regions would become 9 equal subregions
</details>
<br>

# Building from source
* Ensure the following are installed
    * python (>= 3.6)
        * tensorflow == 2.5.0
        * onnx-tensorlfow==1.9.0
    * CMake
    * m4
    * autoconf
    * libtool
    * texlive-latex-base
    * ``` sudo apt install cmake m4 autoconf libtool ```

1. Build and Install ERAN verification engine
    ```
    git clone https://github.com/yodarocks1/eran.git /src/eran/
    cd /src/eran

    ./install.sh                    ~ No gpu acceleration
    ./install.sh --use-cuda         ~ Gpu acceleration

    bash ./gurobi_setup_path.sh
    ENV PYTHONPATH="/src/eran/python_interface"
    ```

2. Install Boost Libraries
    ```
	wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz

    tar -xvzf boost_1_77_0.tar.gz

    cd boost_1_77_0

    ./bootstrap.sh --with-python=python3 --with-libraries=python,system

    ./b2 install
    ```

3. Get and Install VERAPAK
    ```
    mkdir /src/VERAPAK
    cd /src/VERAPAK

    git clone https://github.com/formal-verification-research/VERAPAK.git
    
    mkdir /src/VERAPAK/_build 
    cd /src/VERAPAK/_build

    cmake ..
    make install -j4
    ```

# Contact
Formal Verification Research Lab\
https://formal-verification-research.github.io/
