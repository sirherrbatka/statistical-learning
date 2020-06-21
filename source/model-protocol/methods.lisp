(cl:in-package #:statistical-learning.model-protocol)


(defmethod make-training-state :before ((model fundamental-model)
                                        train-data
                                        target-data
                                        &rest initargs
                                        &key &allow-other-keys)
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


(defmethod predict :before ((model fundamental-model) data &optional parallel)
  (statistical-learning.data:check-data-points data))


(defmethod cl-ds.utils:cloning-information append
    ((object fundamental-training-state))
  `((:training-parameters training-parameters)
    (:weights weights)
    (:target-data target-data)
    (:training-data training-data)))


(defmethod make-training-state ((parameters fundamental-model-parameters)
                                train-data target-data
                                &rest initargs &key &allow-other-keys)
  (apply #'make 'fundamental-training-state
         :training-parameters parameters
         :training-data train-data
         :target-data target-data
         initargs))


(defmethod sample-training-state-info append ((parameters fundamental-model-parameters)
                                              state
                                              &key
                                              data-points
                                              train-attributes
                                              target-attributes
                                              &allow-other-keys)
  (list :training-data (sl.data:sample (sl.mp:training-data state)
                                        :data-points data-points
                                        :attributes train-attributes)
        :target-data (sl.data:sample (sl.mp:target-data state)
                                     :data-points data-points
                                     :attributes target-attributes)
        :weights (if (null (sl.mp:weights state))
                     nil
                     (sl.data:sample (sl.mp:weights state)
                                     :data-points data-points))))


(defmethod sample-training-state* ((parameters fundamental-model-parameters)
                                   state
                                   &key data-points train-attributes target-attributes initargs
                                   &allow-other-keys)
  (let ((cloning-list (cl-ds.utils:cloning-list state))
        (class (class-of state)))
    (apply #'make class
           (append (sample-training-state-info parameters state
                                               :target-attributes target-attributes
                                               :train-attributes train-attributes
                                               :data-points data-points)
                   initargs cloning-list))))
