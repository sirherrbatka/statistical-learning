(cl:in-package #:sl.proxy-tree)


(define-forwarding
  (sl.tp:extract-predictions* (predictions))
  (sl.tp:contribute-predictions* (model data state parallel))
  (sl.tp:make-leaf* (training-state))
  (sl.tp:split* (state leaf))
  (sl.tp:initialize-leaf (state leaf))
  (sl.tp:maximal-depth ())
  (sl.tp:minimal-size ())
  (sl.tp:minimal-difference ())
  (sl.perf:errors (target predictions))
  (sl.tp:trials-count ())
  (sl.perf:average-performance-metric (metrics))
  (sl.tp:parallel ()))


(defmethod sl.perf:performance-metric ((parameters proxy-tree)
                                       target
                                       predictions
                                       &key weights)
  (sl.perf:performance-metric (inner parameters)
                              target
                              predictions
                              :key weights))


(defmethod sl.mp:make-training-state ((parameters honest-tree)
                                      train-data
                                      target-data
                                      &rest initargs
                                      &key attributes &allow-other-keys)
  (cons attributes
        (apply #'call-next-method
               parameters
               train-data
               target-data
               :attributes (~> train-data
                               sl.data:attributes-count
                               sl.data:iota-vector)
               initargs)))


(defmethod sl.mp:make-model* ((parameters honest-tree)
                              attributes.training-state)
  (bind (((attributes . training-state) attributes.training-state)
         (inner (inner parameters))
         ((:values division adjust) (train/adjust attributes training-state))
         (values-training-data (sl.mp:training-data adjust))
         (model (sl.mp:make-model* inner division))
         (root (sl.tp:root model))
         ((:flet assign-leaf (index))
          (cons index
                (sl.tp:leaf-for root values-training-data index)))
         ((:flet adjust-leaf (leaf.indexes))
          (bind (((leaf . indexes) leaf.indexes)
                 (no-fill-pointer (cl-ds.utils:remove-fill-pointer indexes)))
            (~> (sl.mp:sample-training-state adjust
                                             :data-points no-fill-pointer)
                (sl.tp:initialize-leaf inner
                                       _
                                       leaf)))))
    (~> (cl-ds:iota-range :to (sl.data:data-points-count values-training-data))
        (cl-ds.alg:on-each #'assign-leaf)
        (cl-ds.alg:group-by :key #'cdr :test 'eq)
        (cl-ds.alg:to-vector :key #'car :element-type 'fixnum)
        (cl-ds:traverse #'adjust-leaf))
    model))


(defmethod sl.mp:make-model* ((parameters proxy-tree) training-state)
  (lret ((result (sl.mp:make-model* (inner parameters)
                                    training-state)))))


(defmethod sl.mp:make-training-state ((parameters proxy-tree)
                                      train-data
                                      target-data
                                      &rest initargs
                                      &key &allow-other-keys)
  (lret ((result (apply #'sl.mp:make-training-state
                        (inner parameters)
                        train-data target-data initargs)))
    (setf (sl.mp:training-parameters result) parameters)))


(defmethod sl.tp:split-training-state* ((parameters proxy-tree)
                                        state
                                        split-array
                                        position
                                        size
                                        initargs
                                        &optional
                                          attribute-index
                                          attribute-indexes)
  (lret ((result (sl.tp:split-training-state* (inner parameters)
                                              state
                                              split-array
                                              position
                                              size
                                              initargs
                                              attribute-index
                                              attribute-indexes)))
        (setf (sl.mp:training-parameters result) parameters)))


(defmethod sl.mp:sample-training-state* ((parameters proxy-tree)
                                         state
                                         &key data-points
                                           train-attributes
                                           target-attributes
                                           initargs)
  (lret ((result (sl.mp:sample-training-state*
                  (inner parameters)
                  state
                  :data-points data-points
                  :train-attributes train-attributes
                  :target-attributes target-attributes
                  :initargs initargs)))
    (setf (sl.mp:training-parameters result) parameters)))
