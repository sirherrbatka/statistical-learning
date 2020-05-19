(cl:in-package #:cl-grf.performance)


(defmethod performance-metric :before ((parameters cl-grf.mp:fundamental-model-parameters)
                                       target predictions)
  (cl-grf.data:check-data-points target)
  (check-type predictions vector)
  (unless (= (cl-grf.data:data-points-count target)
             (length predictions))
    (error 'cl-ds:incompatible-arguments
           :parameters '(target predictions)
           :values (list target predictions)
           :format-control "TARGET and PREDICITIONS are supposed to contain equal number of data-points")))
