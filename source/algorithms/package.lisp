(cl:in-package #:cl-user)


(defpackage #:statistical-learning.algorithms
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.alg #:sl.alg)
  (:export
   #:basic-regression
   #:calculate-expected-value
   #:calculate-score
   #:classification
   #:gradient-boost-classification
   #:gradient-boost-regression
   #:gradient-boost-response
   #:predictions
   #:regression
   #:score
   #:scored-leaf-node
   #:single-impurity-classification
   #:support))
