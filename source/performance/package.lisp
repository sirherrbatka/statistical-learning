(cl:in-package #:cl-user)


(defpackage #:cl-grf.performance
  (:use #:cl #:cl-grf.aux-package)
  (:export
   #:accuracy
   #:attributes-importance
   #:at-confusion-matrix
   #:average-performance-metric
   #:cross-validation
   #:f1-score
   #:fundamental-model-parameters
   #:make-confusion-matrix
   #:performance-metric
   #:precision
   #:recall
   #:errors
   #:specificity
   #:two-class-confusion-matrix-from-general-confusion-matrix
   ))
