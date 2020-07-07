(cl:in-package #:cl-user)


(defpackage #:statistical-learning.model-protocol
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.mp #:sl.mp)
  (:export
   #:fundamental-model
   #:fundamental-model-parameters
   #:make-model
   #:make-model*
   #:parameters
   #:predict
   #:make-training-state
   #:sample-training-state
   #:sample-training-state-info
   #:sample-training-state*
   #:training-parameters
   #:fundamental-training-state
   #:target-data
   #:train-data
   #:data-points
   #:weights
   ))
