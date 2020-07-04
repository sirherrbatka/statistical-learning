(cl:in-package #:cl-user)


(defpackage #:statistical-learning.ensemble
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.ensemble)
  (:export
   #:ensemble-model
   #:random-forest
   #:dynamic-weights-calculator
   #:static-weights-calculator
   #:update-weights
   #:random-forest-model
   #:gradient-boost-ensemble
   #:gradient-boost-ensemble-model
   #:tree-batch-size
   #:tree-sample-rate
   #:trees))
