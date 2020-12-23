(cl:in-package #:sl.proxy-tree)


(defclass honest-tree (tree-proxy)
  ())


(defclass honest-state (proxy-state)
  ((%attributes :initarg :attributes
                :reader attributes)))


(defmethod cl-ds.utils:cloning-information append ((state honest-state))
  `((:attributes attributes)))


(defmethod sl.mp:sample-training-state*/proxy
    ((proxy honest-tree)
     parameters
     (state honest-state)
     &key data-points
       train-attributes
       target-attributes
       initargs)
  (let* ((inner-sample (sl.mp:sample-training-state*/proxy
                        (sl.common:next-proxy proxy)
                        parameters
                        (inner state)
                        :data-points data-points
                        :target-attributes target-attributes
                        :initargs initargs)))
    (cl-ds.utils:quasi-clone* state
      :inner inner-sample
      :attributes (map '(vector fixnum)
                       (curry #'aref (attributes state))
                       train-attributes))))


(defmethod sl.mp:make-model*/proxy ((proxy honest-tree)
                                    parameters
                                    state)
  (declare (optimize (debug 3)))
  (bind ((inner-state (inner state))
         (training-data (sl.mp:train-data inner-state))
         (data-points-count (~> inner-state
                                sl.mp:data-points
                                length))
         (indexes (sl.data:reshuffle (sl.data:iota-vector data-points-count)))
         (division-indexes (take (truncate data-points-count 2)
                                 indexes))
         (adjust-indexes (drop (truncate data-points-count 2)
                               indexes))
         (attributes (attributes state))
         (next-proxy (sl.common:next-proxy proxy))
         (division (sl.mp:sample-training-state*/proxy
                    next-proxy
                    parameters
                    inner-state
                    :train-attributes attributes
                    :data-points division-indexes))
         (adjust (sl.mp:sample-training-state*/proxy next-proxy
                                                     parameters
                                                     inner-state
                                                     :data-points adjust-indexes))
         (model (sl.mp:make-model*/proxy next-proxy parameters division))
         (root (sl.tp:root model))
         (splitter (sl.tp:splitter parameters))
         ((:flet assign-leaf (index))
          (cons index
                (sl.tp:leaf-for splitter root
                                training-data index
                                model)))
         ((:flet adjust-leaf (leaf.indexes))
          (bind (((leaf . indexes) leaf.indexes)
                 (no-fill-pointer (cl-ds.utils:remove-fill-pointer indexes))
                 (sample (sl.mp:sample-training-state*/proxy
                          next-proxy parameters
                          adjust :data-points no-fill-pointer)))
            (sl.tp:initialize-leaf/proxy next-proxy parameters
                                         sample leaf))))
    (~> (cl-ds.alg:on-each adjust-indexes #'assign-leaf)
        (cl-ds.alg:group-by :key #'cdr :test 'eq)
        (cl-ds.alg:to-vector :key #'car :element-type 'fixnum)
        (cl-ds:traverse #'adjust-leaf))
    model))


(defmethod sl.mp:make-training-state/proxy
    ((proxy honest-tree)
     parameters
     &rest initargs
     &key attributes train-data &allow-other-keys)
  (let ((inner (apply #'sl.mp:make-training-state/proxy
                      (sl.common:next-proxy proxy)
                      parameters
                      :attributes nil
                      initargs)))
    (make 'honest-state
          :training-parameters parameters
          :attributes (or attributes (~> train-data
                                         sl.data:attributes-count
                                         sl.data:iota-vector))
          :inner inner)))


(defun honest (parameters)
  (sl.common:lift parameters 'honest-tree))
