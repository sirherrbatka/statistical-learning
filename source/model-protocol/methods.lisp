(cl:in-package #:cl-grf.model-protocol)


(defmethod make-model :around ((model fundamental-model)
                               train-data
                               target-data)
  (check-type train-data cl-grf.data:data-matrix)
  (check-type target-data cl-grf.data:data-matrix)
  (cl-grf.data:bind-data-matrix-dimensions
      ((train-data-points train-data-attributes train-data)
       (target-data-points target-data-attributes target-data))
    (when (zerop train-data-attributes)
      (error 'cl-ds:invalid-argument
             :value train-data
             :argument 'train-data
             :format-control "TRAIN-DATA has no attributes"))
    (when (zerop train-data-points)
      (error 'cl-ds:invalid-argument
             :value train-data
             :argument 'train-data
             :format-control "TRAIN-DATA has no data-points."))
    (when (zerop target-data-attributes)
      (error 'cl-ds:invalid-argument
             :value target-data
             :argument 'target-data
             :format-control "TARGET-DATA has no attributes"))
    (when (zerop target-data-points)
      (error 'cl-ds:invalid-argument
             :value target-data
             :argument 'target-data
             :format-control "TARGET-DATA has no data-points."))
    (unless (= train-data-points target-data-points)
      (error 'cl-ds:incompatible-arguments
             :arguments '(train-data target-data)
             :values `(,train-data ,target-data)
             :format-control "Number of data-points in the the train-data should be equal to the number of data-points in the target-data."))
    (call-next-method model train-data target-data)))
