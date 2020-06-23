(cl:in-package #:sl.proxy-tree)


(defclass causal-tree (proxy-tree)
  ((%minimal-treatment-size :reader minimal-treatment-size
                            :initarg :minimal-treatment-size)
   (%minimal-no-treatment-size :reader minimal-no-treatment-size
                               :initarg :minimal-no-treatment-size)))


(defclass causal-state (proxy-state)
  ((%treatment :initarg :treatment
               :reader treatment)))


(defclass causal-leaf (sl.tp:fundamental-leaf-node)
  ((%treatment-leaf :initarg :treatment-leaf
                    :accessor treatment-leaf)
   (%no-treatment-leaf :initarg :no-treatment-leaf
                       :accessor no-treatment-leaf))
  (:default-initargs :treatment-leaf nil
                     :no-treatment-leaf nil))


(defmethod initialize-instance :after ((instance causal-tree) &rest initargs)
  (declare (ignore initargs))
  (bind ((minimal-treatment-size (minimal-treatment-size instance))
         (minimal-no-treatment-size (minimal-no-treatment-size instance))
         (minimal-size (sl.tp:minimal-size instance)))
    (check-type minimal-treatment-size integer)
    (check-type minimal-no-treatment-size integer)
    (unless (>= minimal-size (+ minimal-treatment-size
                                minimal-no-treatment-size))
      (error 'cl-ds:incompatible-arguments
             :parameters '(:minimal-size :minimal-no-treatment-size :minimal-treatment-size)
             :values (list minimal-size minimal-no-treatment-size minimal-treatment-size)
             :format-control ":MINIMAL-SIZE must be at least equal to the sum of :MINIMAL-TREATMENT-SIZE and :MINIMAL-NO-TREATMENT-SIZE"))
    (unless (< 0 minimal-no-treatment-size)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :minimal-no-treatment-size
             :bounds '(< 0 :minimal-no-treatment-size)
             :value minimal-no-treatment-size))
    (unless (< 0 minimal-treatment-size)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :minimal-treatment-size
             :bounds '(< 0 :minimal-treatment-size)
             :value minimal-treatment-size))))


(defmethod sl.tp:split-training-state* ((parameters causal-tree)
                                        state
                                        split-array
                                        position
                                        size
                                        initargs
                                        &optional
                                          attribute-index
                                          attribute-indexes)
  (cl-ds.utils:quasi-clone* state
    :treatment (split-treatment (treatment state)
                                size
                                split-array
                                position)
    :inner (sl.tp:split-training-state* (inner parameters)
                                        (inner state)
                                        split-array
                                        position
                                        size
                                        initargs
                                        attribute-index
                                        attribute-indexes)))


(defmethod sl.mp:sample-training-state* ((parameters causal-tree)
                                         state
                                         &key
                                           data-points
                                           train-attributes
                                           initargs
                                           target-attributes)
  (cl-ds.utils:quasi-clone* state
    :treatment (map 'vector
                    (curry #'aref (treatment state))
                    data-points)
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
         (minimal-no-treatment-size (minimal-no-treatment-size training-parameters))
         (minimal-treatment-size (minimal-treatment-size training-parameters))
         (total-size (length treatment))
         (treatment-size (count t treatment))
         (no-treatment-size (- total-size treatment-size)))
    (if (or (< treatment-size (* 2 minimal-treatment-size))
            (< no-treatment-size (* 2 minimal-no-treatment-size)))
        nil
        (call-next-method))))


(defmethod sl.tp:initialize-leaf ((parameters causal-tree)
                                  training-state
                                  leaf)
  (bind ((inner (inner training-state))
         (treatment (treatment training-state))
         (inner-parameters (inner parameters))
         (no-treatment-state (sl.tp:split-training-state* inner-parameters
                                                          inner
                                                          treatment
                                                          nil
                                                          (count nil treatment)
                                                          '()))
         (treatment-state (sl.tp:split-training-state* inner-parameters
                                                       inner
                                                       treatment
                                                       t
                                                       (count t treatment)
                                                       '()))
         (no-treatment-leaf (sl.tp:make-leaf* inner-parameters))
         (treatment-leaf (sl.tp:make-leaf* inner-parameters)))
    (sl.tp:initialize-leaf inner-parameters
                           no-treatment-state
                           no-treatment-leaf)
    (sl.tp:initialize-leaf inner-parameters
                           treatment-state
                           treatment-leaf)
    (setf (treatment-leaf leaf) treatment-leaf
          (no-treatment-leaf leaf) no-treatment-leaf)))


(defclass causal-contributed-predictions ()
  ((%training-parameters :initarg :training-parameters
                         :reader sl.mp:training-parameters)
   (%treatment :initarg :treatment
               :accessor treatment)
   (%no-treatment :initarg :no-treatment
                  :accessor no-treatment))
  (:default-initargs :treatment nil
                     :no-treatment nil))


(defmethod sl.mp:make-training-state ((parameters causal-tree)
                                      &rest initargs
                                      &key treatment data-points &allow-other-keys)
  (make 'causal-state
        :training-parameters parameters
        :inner (apply #'sl.mp:make-training-state
                      (inner parameters)
                      initargs)
        :treatment (map 'vector
                        (compose (curry #'= 1)
                                 (curry #'aref (cl-ds.utils:unfold-table treatment)))
                        data-points)))


(defun causal (parameters
               minimal-treatment-size
               minimal-no-treatment-size)
  (make 'causal-tree
        :minimal-treatment-size minimal-treatment-size
        :minimal-no-treatment-size minimal-no-treatment-size
        :inner parameters))


(defmethod sl.tp:extract-predictions* ((parameters causal-tree)
                                       state)
  (let* ((no-treatment-result (sl.tp:extract-predictions* (inner parameters)
                                                          (no-treatment state)))
         (treatment-result (sl.tp:extract-predictions* (inner parameters)
                                                       (treatment state)))
         (result (sl.data:make-data-matrix-like treatment-result)))
    (iterate
      (for i from 0 below (array-total-size result))
      (setf (row-major-aref result i)
            (- (row-major-aref treatment-result i)
               (row-major-aref no-treatment-result i))))
    result))


;; this is simple, but slow
(defmethod sl.tp:contribute-predictions* ((parameters causal-tree)
                                          model
                                          data
                                          state
                                          parallel
                                          &optional (leaf-key #'identity))
  (when (null state)
    (setf state (make 'causal-contributed-predictions
                      :training-parameters parameters)))
  (setf (treatment state) (sl.tp:contribute-predictions*
                           (inner parameters)
                           model
                           data
                           (treatment state)
                           parallel
                           (compose #'treatment-leaf leaf-key))
        (no-treatment state) (sl.tp:contribute-predictions*
                              (inner parameters)
                              model
                              data
                              (no-treatment state)
                              parallel
                              (compose #'no-treatment-leaf leaf-key)))
  state)


(defmethod sl.mp:make-model* ((parameters causal-tree) state)
  (make 'sl.tp:tree-model
        :parameters parameters
        :root (~>> state sl.tp:make-leaf (sl.tp:split state))))
