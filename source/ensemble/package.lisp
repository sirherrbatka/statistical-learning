(cl:in-package #:cl-user)


(defpackage #:statistical-learning.ensemble
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.ensemble)
  (:export
   #:*state*
   #:after-tree-fitting
   #:after-tree-fitting/proxy
   #:assign-leafs
   #:assigned-leafs
   #:dynamic-weights-calculator
   #:ensemble-model
   #:ensmble-slot
   #:gradient-boost-ensemble
   #:gradient-boost-ensemble-model
   #:indexes
   #:leafs
   #:leafs-assigned-p
   #:make-tree-training-state
   #:make-tree-training-state
   #:make-tree-training-state/proxy
   #:make-tree-training-state/proxy
   #:random-forest
   #:random-forest-model
   #:static-weights-calculator
   #:tree-batch-size
   #:tree-sample-rate
   #:trees
   #:trees-view
   #:update-weights))
