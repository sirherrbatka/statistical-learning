(cl:in-package #:sl.proxy-tree)


(defclass capture-input (tree-proxy)
  ())


(defgeneric captured-input (leaf))


(defclass capture-input-leaf (sl.tp:fundamental-leaf-node)
  ((%inner-leaf :initarg :inner-leaf
                :accessor inner-leaf)
   (%captured-input :initarg :captured-input
                    :accessor captured-input))
  (:default-initargs
   :inner-leaf nil
   :captured-input nil))


(defmethod sl.tp:make-leaf*/proxy ((proxy capture-input)
                                   parameters
                                   state)
  (make 'capture-input-leaf
        :inner-leaf (call-next-method)))


(defmethod sl.tp:initialize-leaf/proxy
    ((proxy capture-input)
     parameters
     training-state
     leaf)
  (call-next-method proxy
                    parameters
                    training-state
                    (inner-leaf leaf))
  (let* ((data (sl.mp:train-data training-state))
         (data-points-count (sl.data:data-points-count data))
         (sums (sl.data:reduce-data-points #'+ data)))
    (iterate
      (for i from 0 below (sl.data:attributes-count sums))
      (setf #1=(sl.data:mref sums 0 i) (/ #1# data-points-count)))
    (setf (captured-input leaf) sums)))


(defmethod sl.tp:predictions ((leaf capture-input-leaf))
  (~> leaf
      inner-leaf
      sl.tp:predictions))


(defmethod (setf sl.tp:predictions) (new-value (leaf capture-input-leaf))
  (setf (sl.tp:predictions (inner-leaf leaf)) new-value))
