(cl:in-package #:cl-user)


(defpackage #:statistical-learning.decision-tree
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.dt #:sl.dt)
  (:export
   #:classification
   #:regression
   ))
