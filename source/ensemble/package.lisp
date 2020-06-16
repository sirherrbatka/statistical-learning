(cl:in-package #:cl-user)


(defpackage #:statistical-learning.ensemble
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.ensemble)
  (:export
   #:ensemble-model
   #:random-forest
   #:random-forest-model
   #:gradient-boost-ensemble
   #:gradient-boost-ensemble-model
   #:tree-batch-size
   #:tree-sample-rate
   #:trees))
