(cl:in-package #:statistical-learning.model-protocol)


(defmethod make-model :before ((model fundamental-model)
                               train-data
                               target-data
                               &key &allow-other-keys)
  (statistical-learning.data:bind-data-matrix-dimensions
      ((train-data-points train-data-attributes train-data)
       (target-data-points target-data-attributes target-data))
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
  (statistical-learning.data:check-data-points data))
