(cl:in-package #:sl.proxy-tree)


(defclass causal-tree (proxy-tree)
  ((%minimal-treatment-size :reader minimal-treatment-size
                            :initarg :minimal-treatment-size)
   (%treatment-types-count :reader treatment-types-count
                           :initarg :treatment-types-count))
  (:default-initargs
   :treatment-types-count 2))


(defclass causal-state (proxy-state)
  ((%treatment :initarg :treatment
               :reader treatment)))


(defmethod cl-ds.utils:cloning-information append ((object causal-state))
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
         (treatment-types-count (treatment-types-count instance))
         (minimal-size (sl.tp:minimal-size instance)))
    (check-type minimal-treatment-size integer)
    (check-type treatment-types-count integer)
    (unless (>= minimal-size (* 2 minimal-treatment-size))
      (error 'cl-ds:incompatible-arguments
             :parameters '(:minimal-size :minimal-no-treatment-size :minimal-treatment-size)
             :values (list minimal-size minimal-treatment-size)
             :format-control ":MINIMAL-SIZE must be at least equal to the sum of :MINIMAL-TREATMENT-SIZE and :MINIMAL-NO-TREATMENT-SIZE"))
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


(defmethod sl.tp:split-training-state* ((parameters causal-tree)
                                        state
                                        split-array
                                        position
                                        size
                                        initargs
                                        point)
  (cl-ds.utils:quasi-clone* state
    :inner (sl.tp:split-training-state* (inner parameters)
                                        (inner state)
                                        split-array
                                        position
                                        size
                                        initargs
                                        point)))


(defmethod sl.mp:sample-training-state* ((parameters causal-tree)
                                         state
                                         &key
                                           data-points
                                           train-attributes
                                           initargs
                                           target-attributes)
  (cl-ds.utils:quasi-clone* state
    :inner (sl.mp:sample-training-state* (inner parameters)
                                         (inner state)
                                         :data-points data-points
                                         :train-attributes train-attributes
                                         :initargs initargs
                                         :target-attributes target-attributes)))


(defmethod sl.tp:make-leaf* ((parameters causal-tree))
  (make 'causal-leaf))


(defmethod sl.tp:split* :around ((training-parameters causal-tree)
                                 training-state)
  (let* ((treatment (treatment training-state))
         (data-points (sl.tp:data-points training-state))
         (minimal-treatment-size (minimal-treatment-size training-parameters))
         (treatment-frequency (make-hash-table)))
    (iterate
      (for i in-vector data-points)
      (incf (gethash (svref treatment i) treatment-frequency 0)))
    (iterate
      (for (key count) in-hashtable treatment-frequency)
      (when (< count (* 2 minimal-treatment-size))
        (leave nil))
      (finally (return (call-next-method))))))


(defmethod sl.tp:initialize-leaf ((parameters causal-tree)
                                  training-state
                                  leaf)
  (bind ((inner (inner training-state))
         (treatment (treatment training-state))
         (inner-parameters (inner parameters))
         (data-points (sl.tp:data-points inner))
         (treatment-types-count (treatment-types-count parameters))
         (leafs (make-array treatment-types-count))
         (sizes (make-array treatment-types-count))
         (treatment-vector (map 'vector
                                (curry #'aref treatment)
                                data-points))
         ((:flet treatment-size (i))
          (count i treatment-vector)))
    (iterate
      (for i from 0 below treatment-types-count)
      (for sub-leaf = (sl.tp:make-leaf* inner-parameters))
      (for treatment-size = (treatment-size i))
      (for treatment-state = (sl.tp:split-training-state* inner-parameters
                                                          inner
                                                          treatment-vector
                                                          i
                                                          treatment-size
                                                          '()
                                                          nil))
      (sl.tp:initialize-leaf inner-parameters
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


(defmethod sl.mp:make-training-state ((parameters causal-tree)
                                      &rest initargs
                                      &key treatment &allow-other-keys)
  (make 'causal-state
        :training-parameters parameters
        :inner (apply #'sl.mp:make-training-state
                      (inner parameters)
                      initargs)
        :treatment (map 'vector
                        #'round
                        (cl-ds.utils:unfold-table treatment))))


(defun causal (parameters
               minimal-treatment-size
               treatment-classes)
  (make 'causal-tree
        :minimal-treatment-size minimal-treatment-size
        :treatment-types-count treatment-classes
        :inner parameters))


(defmethod sl.tp:extract-predictions* ((parameters causal-tree)
                                       state)
  (map 'vector
       (curry #'sl.tp:extract-predictions* (inner parameters))
       (results state)))


;; this is simple, but slow
(defmethod sl.tp:contribute-predictions* ((parameters causal-tree)
                                          model
                                          data
                                          state
                                          parallel
                                          &optional (leaf-key #'identity))
  (let ((treatment-types-count (treatment-types-count parameters)))
    (when (null state)
      (setf state (make 'causal-contributed-predictions
                        :results (make-array treatment-types-count
                                             :initial-element nil)
                        :training-parameters parameters)))
    (iterate
      (with inner = (inner parameters))
      (with results = (results state))
      (for i from 0 below (treatment-types-count parameters))
      (setf (aref results i) (sl.tp:contribute-predictions*
                              inner
                              model
                              data
                              (aref results i)
                              parallel
                              (compose (rcurry #'aref i)
                                       #'leafs
                                       leaf-key)))))
  state)


(defmethod sl.mp:make-model* ((parameters causal-tree) state)
  (make 'sl.tp:tree-model
        :parameters parameters
        :root (~>> state sl.tp:make-leaf (sl.tp:split state))))


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
