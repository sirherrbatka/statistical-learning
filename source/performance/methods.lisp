(cl:in-package #:statistical-learning.performance)


(defmethod performance-metric :before ((parameters statistical-learning.mp:fundamental-model-parameters)
                                       target predictions)
  (statistical-learning.data:check-data-points target predictions))
