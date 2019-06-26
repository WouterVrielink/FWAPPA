# Fireworks versus Plant Propagation

This repo holds the codebase the ['Fireworks Algorithm versus Plant Propagation Algorithm' studies](TODO). From the abstract:

*In recent years, the field of Evolutionary Algorithms has seen a tremendous increase in novel methods. While these algorithmic innovations often show excellent results on relatively limited domains, they are less often rigorously cross-tested or compared to other state-of-the-art developments. Two of these methods, quite similar in their appearance, are the Fireworks Algorithm and Plant Propagation Algorithm.

This study compares the similarities and differences between these two algorithms, from both quantitative and qualitative perspectives, by comparing them on a set of carefully chosen benchmark test functions. The Fireworks Algorithm outperforms the Plant Propagation Algorithm on the majority of these, but when the functions are shifted slightly, Plant Propagation gives better results. Reasons behind these surprising differences are presented, comparison methods for evolutionary algorithms are discussed in a wider context, and novel methods that can be used to detect locational and boundary biases are explored. All source code, graphs, test functions, and algorithmic implementations have been made publicly available for reference and further reuse.*

The idea for this project came from the papers by [Ying Tan and Yuanchun Zhu (2010)](https://www.researchgate.net/profile/Ying_Tan5/publication/220704568_Fireworks_Algorithm_for_Optimization/links/00b7d5281fc26a092a000000.pdf) and [Abdellah Salhi and Eric S Fraga (2011)](http://repository.essex.ac.uk/9974/1/paper.pdf) who both presented their algorithms at META16 in Marrakesh. The algorithms are very similar in their main principles, but their subroutines differ greatly.

This project was conducted as a master thesis for the study Computational Science at the University of Amsterdam.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Prerequisite libraries, frameworks, and modules can be installed through:

```
pip install -r requirements.txt
```

This will install the correct versions of:
- matplotlib (3.0.3)
- pandas (0.24.2)
- scipy (1.1.0)
- numpy (1.15.4)
- seaborn (0.9.0)

### Repository
The following list describes the most important files in the project and where to find them:
- **/Code**: contains all of the codebase of this project.
  - **/Code/batchrunner.py**: contains all the code required to run all experiments.
  - **/Code/benchmark_functions.py**: contains all the benchmark functions.
  - **/Code/benchmarks.py**: contains the Benchmark class, the set_benchmark_properties decorator,
  the param_shift helper function, and the apply_add function.
  - **/Code/environment.py**: contains the Environment class.
  - **/Code/fireworks.py**: contains the code of the Fireworks algorithm.
  - **/Code/helper_tools.py**: contains all the code required to perform statistics and generate graphs.
  - **/Code/plantpropagation.py**: contains the code of the Plant Propagation algorithm.
  - **/Code/point.py**: contains a coordinate class that can calculate distances and determine if
    it was already once evaluated.
- **/Configs**: contains the configuration files for the algorithms (JSON).

To get an idea of how the code works and how to run your own experiments, please take a look at the `__main__` function in `batchrunner.py`. To get an idea of how to perform statistical analysis and how to create graphs, please take a look at the `__main__` function in `helper_tools.py`.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/WouterVrielink/FWAPPA/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **W. L. J. Vrielink** - *Initial work*

See also the list of [contributors](https://github.com/WouterVrielink/FWAPPA/graphs/contributors) who participated in this project.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](https://github.com/WouterVrielink/FWAPPA/LICENSE.md) file for details

## Acknowledgments

I would like to thank Marcus Pfundstein (former student, UvA) for his preliminary work compiling a set of benchmarks, Quinten van der Post (colleague, UvA) for allowing me to use his server as a computing platform -- and filling it to the brim with results--, Hans-Paul Schwefel (Professor Emeritus, University of Dortmund) for answering our questions about the Schwefel benchmark function by email, and Ying Tan (Professor, Peking University), Abdellah Salhi (Professor, University of Essex), and Eric Fraga (Professor, UCL London) for answering our questions on FWA and PPA respectively. 

Further appreciation goes out to Rick Quax (Assistant Professor, UvA), without whom the idea of bias detection would not have come to fruition. Finally, deep gratitude goes out to Daan van den Berg (Doctorandus, UvA) who helped throughout the entire course of the thesis, proofreading several times and spending countless hours discussing results and the algorithms.
