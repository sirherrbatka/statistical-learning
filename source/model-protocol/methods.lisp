(cl:in-package #:cl-grf.model-protocol)


(defmethod make-model :before ((model fundamental-model)
                               train-data
                               target-data
                               &optional weights)
  (if (null weights)
      (cl-grf.data:check-data-points train-data target-data)
      (cl-grf.data:check-data-points train-data target-data weights))
  (cl-grf.data:bind-data-matrix-dimensions
      ((train-data-points train-data-attributes train-data)
       (target-data-points target-data-attributes target-data))
    (when (and weights
               (not (= 1 (cl-grf.data:attributes-count weights))))
      (error 'cl-ds:invalid-argument-value
             :value weights
             :argument 'weights
             :format-control "Weights data-matrix is supposed to have exactly one attribute."))
    (when (zerop train-data-attributes)
      (error 'cl-ds:invalid-argument-value
             :value train-data
             :argument 'train-data
             :format-control "TRAIN-DATA has no attributes"))
    (when (zerop train-data-points)
      (error 'cl-ds:invalid-argument-value
             :value train-data
             :argument 'train-data
             :format-control "TRAIN-DATA has no data-points."))
    (when (zerop target-data-attributes)
      (error 'cl-ds:invalid-argument-value
             :value target-data
             :argument 'target-data
             :format-control "TARGET-DATA has no attributes"))
    (when (zerop target-data-points)
      (error 'cl-ds:invalid-argument-value
             :value target-data
             :argument 'target-data
             :format-control "TARGET-DATA has no data-points."))))


(defmethod predict :before ((model fundamental-model) data &optional parallel)
  (cl-grf.data:check-data-points data))
