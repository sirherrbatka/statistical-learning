(cl:in-package #:cl-user)


(defpackage #:statistical-learning.model-protocol
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.mp #:sl.mp)
  (:export
   #:data-points
   #:fundamental-model
   #:fundamental-model-parameters
   #:fundamental-training-state
   #:make-model*
   #:make-model*/proxy
   #:make-supervised-model
   #:make-training-state
   #:make-training-state/proxy
   #:make-unsupervised-model
   #:parameters
   #:predict
   #:sample-training-state
   #:sample-training-state*
   #:sample-training-state*/proxy
   #:sample-training-state-info
   #:sample-training-state-info/proxy
   #:supervised-model
   #:target-data
   #:train-data
   #:training-parameters
   #:weights
   #:cache
   ))
