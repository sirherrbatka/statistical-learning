(cl:in-package #:cl-user)


(defpackage #:statistical-learning.ensemble
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.ensemble)
  (:export
   #:dynamic-weights-calculator
   #:ensemble-model
   #:gradient-boost-ensemble
   #:gradient-boost-ensemble-model
   #:leafs
   #:make-tree-training-state
   #:make-tree-training-state/proxy
   #:*state*
   #:random-forest
   #:random-forest-model
   #:static-weights-calculator
   #:tree-batch-size
   #:tree-sample-rate
   #:trees
   #:update-weights))
