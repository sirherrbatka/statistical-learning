(cl:in-package #:cl-user)


(defpackage #:cl-grf.algorithms
  (:use #:cl #:cl-grf.aux-package)
  (:nicknames #:cl-grf.alg)
  (:export
   #:information-gain-classification
   #:support
   #:calculate-score
   #:predictions
   #:support
   #:score
   #:scored-tree-node
   #:scored-leaf-node))
