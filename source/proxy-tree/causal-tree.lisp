(cl:in-package #:sl.proxy-tree)


(defclass causal-tree (tree-proxy)
  ((%minimal-treatment-size :reader minimal-treatment-size
                            :initarg :minimal-treatment-size)
   (%treatment-types-count :reader treatment-types-count
                           :initarg :treatment-types-count))
  (:default-initargs
   :treatment-types-count 2))


(defclass causal-state (proxy-state)
  ((%treatment :initarg :treatment
               :reader treatment)))


(defmethod cl-ds.utils:cloning-information append
    ((object causal-state))
  '((:treatment treatment)))


(defclass causal-leaf (sl.tp:fundamental-leaf-node)
  ((%leafs :initarg :leafs
           :accessor leafs)
   (%sizes :initarg :sizes
           :accessor sizes))
  (:default-initargs
   :leafs nil
   :sizes nil))


(defmethod initialize-instance :after ((instance causal-tree) &rest initargs)
  (declare (ignore initargs))
  (bind ((minimal-treatment-size (minimal-treatment-size instance))
         (treatment-types-count (treatment-types-count instance)))
    (check-type minimal-treatment-size integer)
    (check-type treatment-types-count integer)
    (unless (> treatment-types-count 1)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :treatment-types-count
             :value treatment-types-count
             :bounds '(> :treatment-types-count 0)))
    (unless (> minimal-treatment-size 0)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :minimal-treatment-size
             :bounds '(> :minimal-treatment-size 0)
             :value minimal-treatment-size))))


(defmethod sl.tp:split-training-state*/proxy
    ((proxy causal-tree)
     parameters
     (state causal-state)
     split-array
     position size initargs point)
  (cl-ds.utils:quasi-clone
   state
   :inner (sl.tp:split-training-state*/proxy
           (sl.common:next-proxy proxy)
           parameters
           (inner state)
           split-array
           position
           size
           initargs
           point)))


(defmethod sl.mp:sample-training-state*/proxy
    ((proxy causal-tree)
     parameters
     state
     &key
       data-points
       train-attributes
       initargs
       target-attributes)
  (cl-ds.utils:quasi-clone* state
    :inner (sl.mp:sample-training-state*/proxy
            (sl.common:next-proxy proxy)
            parameters
            (inner state)
            :initargs initargs
            :data-points data-points
            :train-attributes train-attributes
            :target-attributes target-attributes)))


(defmethod sl.tp:make-leaf*/proxy ((proxy causal-tree)
                                   parameters
                                   state)
  (make 'causal-leaf))


(defmethod sl.tp:requires-split-p/proxy
    and ((proxy causal-tree)
         splitter
         training-parameters
         training-state)
  (let* ((treatment (treatment training-state))
         (minimal-treatment-size (minimal-treatment-size proxy))
         (treatment-frequency (make-hash-table)))
    (iterate
      (for i from 0 below (sl.data:data-points-count treatment))
      (incf (gethash (svref treatment i) treatment-frequency 0)))
    (iterate
      (for (key count) in-hashtable treatment-frequency)
      (when (< count (* 2 minimal-treatment-size))
        (leave nil))
      (finally (return t)))))


(defmethod sl.tp:initialize-leaf/proxy
    ((proxy causal-tree)
     parameters
     training-state
     leaf)
  (bind ((inner (inner training-state))
         (treatment (treatment training-state))
         (treatment-types-count (treatment-types-count proxy))
         (leafs (make-array treatment-types-count))
         (sizes (make-array treatment-types-count))
         (next-proxy (sl.common:next-proxy proxy))
         (inner-data (sl.data:data treatment))
         (treatment-vector (~> treatment sl.data:data-points-count make-array
                               (map-into (lambda (index) (aref inner-data index 0))
                                         (sl.data:index treatment))))
         ((:flet treatment-size (i))
          (count i treatment-vector)))
    (iterate
      (for i from 0 below treatment-types-count)
      (for sub-leaf = (sl.tp:make-leaf*/proxy next-proxy parameters training-state))
      (for treatment-size = (treatment-size i))
      (for treatment-state = (sl.tp:split-training-state*/proxy next-proxy
                                                                parameters
                                                                inner
                                                                treatment-vector
                                                                i
                                                                treatment-size
                                                                '()
                                                                nil))
      (sl.tp:initialize-leaf/proxy next-proxy
                                   parameters
                                   treatment-state
                                   sub-leaf)
      (setf (aref sizes i) treatment-size
            (aref leafs i) sub-leaf))
    (setf (leafs leaf) leafs
          (sizes leaf) sizes)))


(defclass causal-contributed-predictions ()
  ((%training-parameters :initarg :training-parameters
                         :reader sl.mp:training-parameters)
   (%results :initarg :results
             :reader results)))


(defmethod sl.mp:make-training-state/proxy
    ((proxy causal-tree)
     parameters
     &rest initargs
     &key treatment &allow-other-keys)
  (make-instance 'causal-state
                 :training-parameters parameters
                 :inner (apply #'sl.mp:make-training-state/proxy
                               (sl.common:next-proxy proxy)
                               parameters
                               initargs)
                 :treatment (map 'vector
                                 #'round
                                 (cl-ds.utils:unfold-table treatment))))


(defun causal (parameters
               minimal-treatment-size
               treatment-classes)
  (sl.common:lift parameters 'causal-tree
                  :minimal-treatment-size minimal-treatment-size
                  :treatment-types-count treatment-classes))


(defmethod sl.tp:extract-predictions*/proxy ((proxy causal-tree)
                                             parameters
                                             state)
  (map 'vector
       (curry #'sl.tp:extract-predictions*/proxy
              (sl.common:next-proxy proxy)
              parameters)
       (results state)))


;; this is simple, but slow
(defmethod sl.tp:contribute-predictions*/proxy
    ((proxy causal-tree)
     parameters
     model
     data
     state
     context
     parallel
     &optional (leaf-key #'identity))
  (ensure leaf-key #'identity)
  (let ((treatment-types-count (treatment-types-count proxy))
        (next-proxy (sl.common:next-proxy proxy)))
    (when (null state)
      (setf state (make 'causal-contributed-predictions
                        :results (make-array treatment-types-count
                                             :initial-element nil)
                        :training-parameters parameters)))
    (iterate
      (with results = (results state))
      (for i from 0 below treatment-types-count)
      (setf (aref results i) (sl.tp:contribute-predictions*/proxy
                              next-proxy
                              parameters
                              model
                              data
                              (aref results i)
                              context
                              parallel
                              (compose (rcurry #'aref i)
                                       #'leafs
                                       leaf-key)))))
  state)


(defmethod sl.mp:make-model*/proxy ((proxy causal-tree) parameters state)
  (make 'sl.tp:tree-model
        :parameters parameters
        :root (sl.tp:make-tree state)))


(defmethod sl.tp:predictions ((leaf causal-leaf))
  (iterate
    (with leafs = (leafs leaf))
    (with sizes = (sizes leaf))
    (with total-size = (reduce #'+ sizes))
    (with count = (length sizes))
    (with result = nil)
    (for i from 0 below count)
    (for leaf = (svref leafs i))
    (for size = (svref sizes i))
    (for predictions = (sl.tp:predictions leaf))
    (ensure result
      (sl.data:make-data-matrix-like predictions))
    (iterate
      (for i from 0 below (array-total-size predictions))
      (incf (row-major-aref result i)
            (* (/ size total-size)
               (row-major-aref predictions i))))
    (finally (return result))))
