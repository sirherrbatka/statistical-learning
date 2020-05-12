(cl:in-package #:cl-user)


(defpackage #:cl-grf.model-protocol
  (:use #:cl #:cl-grf.aux-package)
  (:nicknames #:cl-grf.mp)
  (:export
   #:make-model
   #:fundamental-model
   #:predict
   #:average-performance-metric
   #:fundamental-model-parameters
   #:fundamental-performance-metric
   #:performance-metric
   #:confusion-matrix
   #:total
   #:false-negative
   #:false-positive
   #:true-positive
   #:false-positive
   #:positive
   #:ositive
   #:egative
   #:negative
   #:positive
   #:true
   #:false
   #:negative
   #:precision
   #:f1-score
   #:accuracy
   #:reacll
   #:specificity))
