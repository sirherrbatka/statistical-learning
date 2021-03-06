(cl:in-package #:statistical-learning.model-protocol)


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
  `((:training-parameters training-parameters)))


(defmethod predict :before ((model fundamental-model) data &optional parallel)
  (declare (ignore parallel))
  (statistical-learning.data:check-data-points data))


(defmethod make-training-state/proxy
    (parameters/proxy
     (parameters fundamental-model-parameters)
     &rest initargs &key &allow-other-keys)
  (apply #'make 'fundamental-training-state
         :training-parameters parameters
         initargs))


(defmethod sample-training-state*/proxy
    (parameters/proxy
     (parameters fundamental-model-parameters)
     state
     &key data-points train-attributes target-attributes initargs
     &allow-other-keys)
  (apply #'make (class-of state)
         (append (sample-training-state-info
                  parameters state
                  :target-attributes target-attributes
                  :train-attributes train-attributes
                  :data-points data-points)
                 initargs
                 (cl-ds.utils:cloning-list state))))


(defmethod cache ((state fundamental-training-state)
                  key)
  (gethash key (cached state)))


(defmethod (setf cache) (new-value
                         (state fundamental-training-state)
                         key)
  (setf (gethash key (cached state)) new-value))
