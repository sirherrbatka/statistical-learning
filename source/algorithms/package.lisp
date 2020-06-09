(cl:in-package #:cl-user)


(defpackage #:cl-grf.algorithms
  (:use #:cl #:cl-grf.aux-package)
  (:nicknames #:cl-grf.alg)
  (:export
   #:support
   #:single-impurity-classification
   #:calculate-score
   #:regression
   #:predictions
   #:classification
   #:support
   #:score
   #:scored-tree-node
   #:scored-leaf-node))
