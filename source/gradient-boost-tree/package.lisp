(cl:in-package #:cl-user)


(defpackage #:statistical-learning.gradient-boost-tree
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.gbt #:sl.gbt)
  (:export
   #:classification
   #:regression
   #:implementation
   #:target
   #:calculate-expected-value
   #:calculate-response))
