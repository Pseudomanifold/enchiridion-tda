# An enchiridion for topological data analysis

Following the tradition of providing an enchiridion to master unknown
subjects, this repository contains an introduction into the dark art,
viz. *topological data analysis*.

[Here](Slides.pdf) are the slides that were shown on the [Basel Postdoc Retreat
2018](https://postdocretreat.biozentrum.unibas.ch).

To reproduce the results, please use the script `svm.py`. This requires
a working installation of `numpy` and `scikit-learn`. To run the script
for a given data set&nbsp;(see the `Data` folder for selected data sets
that are shipped with this repository), please run the following:

    $ ./Scripts/svm.py /Results/MUTAG

This should result in the following output:

    INFO:root:Processing Results/MUTAG...
    INFO:root:Setting parameters for data set MUTAG to {'C': 1000.0}
    ---
    MUTAG
    ---
    Accuracy             : 0.95
    Classification report:
                 precision    recall  f1-score   support

              0       0.86      1.00      0.92         6
              1       1.00      0.92      0.96        13

    avg / total       0.95      0.95      0.95        19

The script can also run the analysis for multiple directories.

# Methods

There are several repositories available for topological data analysis:

* [Aleph](https://github.com/Submanifold/Aleph), which is maintained
  by the author of this repository and members of the [Borgwardt
  Lab](https://github.com/BorgwardtLab). It is also used to obtain
  the kernel matrices that are stored in this repository.
* [Dionysus 2](http://mrzv.org/software/dionysus2), developed by [Dmitriy Morozov](http://www.mrzv.org).
* [Ripsers](https://github.com/Ripser/ripser), a lean C++ implementation
  of the Vietoris&ndash;Rips complex construction by [Ulrich
  Bauer](http://ulrich-bauer.org).
* [The topology toolkit&nbsp;(TTK)](https://topology-tool-kit.github.io), which is
  developed by, among others, [Julien Tierny](https://www-pequan.lip6.fr/~tierny).

For *graph kernels*, a general class of methods for analysing graphs
with various properties, I recommend the [`graphkernels`](https://github.com/BorgwardtLab/graphkernels) package,
which offers high-quality and thoroughly-tested implementations of
numerous graph kernels. For more information, please [read the
accompanying
paper](https://academic.oup.com/bioinformatics/article/34/3/530/4209994).

# Disclaimer

No manifolds were harmed in the creation of this repository.
