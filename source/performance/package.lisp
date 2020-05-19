(cl:in-package #:cl-user)


(defpackage #:cl-grf.performance
  (:use #:cl #:cl-grf.aux-package)
  (:export
   #:accuracy
   #:at-confusion-matrix
   #:average-performance-metric
   #:cross-validation
   #:f1-score
   #:fundamental-model-parameters
   #:make-confusion-matrix
   #:performance-metric
   #:precision
   #:recall
   #:specificity
   #:two-class-confusion-matrix-from-general-confusion-matrix
   ))
