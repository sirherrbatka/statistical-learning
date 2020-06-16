(cl:in-package #:cl-user)


(defpackage #:statistical-learning.optimization
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.opt #:sl.opt)
  (:export
   #:<squared-error>
   #:fundamental-loss-function
   #:gini-impurity
   #:gini-impurity-function
   #:k-logistic
   #:k-logistic-function
   #:left
   #:loss
   #:make-split-array
   #:make-split-array
   #:number-of-classes
   #:response
   #:right
   #:split-array
   #:squared-error
   #:squared-error-function
   #:weights-data-matrix
   ))
