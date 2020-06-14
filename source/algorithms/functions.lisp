(cl:in-package #:cl-grf.algorithms)


(defun gradient-boost-response (gathered-predictions expected)
  (gradient-boost-response*
   (cl-grf.tp:training-parameters gathered-predictions)
   expected
   gathered-predictions))
