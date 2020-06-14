(cl:in-package #:cl-user)


(defpackage #:statistical-learning.ensemble
  (:use #:cl #:statistical-learning.aux-package)
  (:export
   #:ensemble
   #:random-forest-parameters
   #:gradient-boost-ensemble
   #:gradient-boost-ensemble-parameters
   #:tree-batch-size
   #:tree-sample-rate
   #:trees))
