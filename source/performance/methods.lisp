(cl:in-package #:cl-grf.performance)


(defmethod performance-metric :before ((parameters cl-grf.mp:fundamental-model-parameters)
                                       target predictions)
  (cl-grf.data:check-data-points target predictions))
