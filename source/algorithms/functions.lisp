(cl:in-package #:statistical-learning.algorithms)


(defun gradient-boost-response (gathered-predictions expected)
  (gradient-boost-response*
   (statistical-learning.tp:training-parameters gathered-predictions)
   expected
   gathered-predictions))
