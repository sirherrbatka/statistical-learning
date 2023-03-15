(cl:in-package #:statistical-learning.model-protocol)


(defmethod cl-ds.utils:cloning-information append ((object fundamental-model))
  '((:parameters parameters)))


(defmethod make-training-state/proxy
    :before (parameters/proxy
             (parameters supervised-model)
             &rest initargs
             &key train-data target-data &allow-other-keys)
  (declare (ignore initargs))
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


(defmethod cl-ds.utils:cloning-information append
    ((object fundamental-training-state))
  `((:training-parameters training-parameters)
    (:model-cached model-cached)))


(defmethod make-training-state/proxy
    (parameters/proxy
     (parameters fundamental-model-parameters)
     &rest initargs &key &allow-other-keys)
  (apply #'make 'fundamental-training-state
         :training-parameters parameters
         initargs))


(defmethod cache ((state fundamental-training-state)
                  key)
  (gethash key (cached state)))


(defmethod (setf cache) (new-value
                         (state fundamental-training-state)
                         key)
  (setf (gethash key (cached state)) new-value))


(defmethod model-cache ((state fundamental-training-state)
                  key)
  (gethash key (model-cached state)))


(defmethod (setf model-cache) (new-value
                               (state fundamental-training-state)
                               key)
  (setf (gethash key (model-cached state)) new-value))
