ReliefF-LSH
=====================================================

Distributed approximation of the feature ranking obtained by ReliefF for Big Data using Locality-Sensitive Hashing.

## Usage: 

        Usage: ReliefFFeatureSelector dataset [options]
                Dataset must be a libsvm or text file
            Options:
                -t    Attribute types. String consisting of N or C for each attribute
                -ct    Class type. Either N (numerical) or C (categorical)  
                -k    Number of neighbors (default: 10)
                -m    Method used to compute the graph. Valid values: lsh, brute (default: lsh)
                -r    Starting radius (default: 0.1)
                -c    Maximum comparisons per item (default: auto)
                -p    Number of partitions for the data RDDs (default: 3*sc.defaultParallelism)
                -s    Skip graph refinement (only LSH) (default: false)
            
            Advanced LSH options:
                -n    Number of hashes per item (default: auto)
                -l    Hash length (default: auto)

## Contributors

- Carlos Eiras-Franco (carlos.eiras.franco@udc.es)

##References


