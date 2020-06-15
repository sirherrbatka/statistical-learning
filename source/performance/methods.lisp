(cl:in-package #:statistical-learning.performance)


(defmethod performance-metric :before ((parameters sl.mp:fundamental-model-parameters)
                                       target predictions
                                       &key weights)
  (check-type weights (or null (simple-array double-float (*))))
  (statistical-learning.data:check-data-points target predictions))
