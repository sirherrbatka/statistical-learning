(cl:in-package #:sl.ct)


(defmethod sl.mp:sample-training-state-info append ((parameters causal)
                                                    state
                                                    &key data-points
                                                    &allow-other-keys)
  (list :treatment (sl.data:sample (treatment state)
                                   :data-points data-points)))


(defmethod sl.tp:split-training-state-info append ((parameters causal)
                                                   state
                                                   split-array
                                                   position
                                                   size
                                                   &optional attribute-index
                                                     attribute-indexes)
  (declare (ignore attribute-index attribute-indexes))
  (list :treatment (sl.data:split (treatment state)
                                  size
                                  split-array
                                  position
                                  nil)))


(defmethod cl-ds.utils:cloning-information append ((state causal-tree-training-state))
  `((:treatment treatment)))


(defmethod initialize-instance :after ((instance causal) &rest initargs)
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


(defmethod sl.tp:split* :around ((parameters causal)
                                 training-state
                                 leaf)
  (bind ((minimal-treatment-size (* 2 (minimal-treatment-size parameters)))
         (minimal-no-treatment-size (* 2 (minimal-no-treatment-size parameters)))
         (treatment (treatment training-state)))
    (iterate
      (with treatment-count = 0)
      (with no-treatment-count = 0)
      (with data-points-count = (sl.data:data-points-count treatment))
      (for i from 0 below data-points-count)
      (if (zerop (sl.data:mref treatment i 0))
          (incf treatment-count)
          (incf no-treatment-count))
      (when (and (>= treatment-count minimal-treatment-size)
                 (>= no-treatment-count minimal-no-treatment-size))
        (leave (call-next-method)))
      (finally (return nil)))))


(defmethod sl.tp:initialize-leaf ((parameters causal-regression)
                                  state
                                  leaf)
  (bind ((predictions (sl.data:make-data-matrix 2 1))
         (target-data (sl.mp:target-data state))
         (treatment (sl.mp:target-data state))
         (support (sl.data:data-points-count target-data)))
    (iterate
      (for i from 0 below support)
      (incf (sl.data:mref predictions
                          (truncate (sl.data:mref treatment i 0))
                          0)
            (sl.data:mref target-data i 0)))
    (setf (sl.tp:predictions leaf) predictions
          (sl.tp:support leaf) support)))


(defmethod sl.tp:initialize-leaf ((parameters causal-regression)
                                  state
                                  leaf)
  (bind ((predictions (make-array 2 :element-type 'double-float
                                    :initial-element 0.0d0))
         (target-data (sl.mp:target-data state))
         (treatments (sl.mp:target-data state))
         (support (sl.data:data-points-count target-data))
         (no-treatment-count 0)
         (treatment-count 0))
    (iterate
      (for i from 0 below support)
      (for treatment = (truncate (sl.data:mref treatments i 0)))
      (if (zerop treatment)
          (incf no-treatment-count)
          (incf treatment-count))
      (incf (aref predictions treatment)
            (sl.data:mref target-data i 0)))
    (setf #1=(aref predictions 0) (/ #1# no-treatment-count)
          #2=(aref predictions 1) (/ #2# treatment-count)
          (sl.tp:predictions leaf) predictions
          (sl.tp:support leaf) support)))


(defmethod sl.tp:initialize-leaf ((parameters causal-classificaton)
                                  state
                                  leaf)
  (bind ((number-of-classes (number-of-classes parameters))
         (predictions (sl.data:make-data-matrix 2 number-of-classes))
         (target-data (sl.mp:target-data state))
         (treatments (sl.mp:target-data state))
         (support (sl.data:data-points-count target-data))
         (no-treatment-count 0)
         (treatment-count 0))
    (iterate
      (for i from 0 below support)
      (for treatment = (truncate (sl.data:mref treatments i 0)))
      (for class = (truncate (sl.data:mref target-data i 0)))
      (if (zerop treatment)
          (incf no-treatment-count)
          (incf treatment-count))
      (incf (sl.data:mref predictions treatment class)))
    (iterate
      (for i from 0 below number-of-classes)
      (setf #1=(sl.data:mref predictions 0 i) (/ #1# no-treatment-count)
            #2=(sl.data:mref predictions 1 i) (/ #2# treatment-count)))
    (setf (sl.tp:predictions leaf) predictions
          (sl.tp:support leaf) support)))


(defmethod sl.tp:contribute-predictions* ((parameters causal-regression)
                                          model
                                          data
                                          state
                                          parallel)
  (let* ((data-points-count (sl.data:data-points-count data)))
    (when (null state)
      (setf state (make-instance 'sl.tp:contributed-predictions
                                 :training-parameters parameters
                                 :sums (sl.data:make-data-matrix data-points-count
                                                                 2)
                                 :indexes (sl.data:iota-vector data-points-count))))
    (let ((sums (sl.tp:sums state))
          (root (sl.tp:root model)))
      (funcall (if parallel #'lparallel:pmap #'map)
               (lambda (index)
                 (let* ((leaf (sl.tp:leaf-for root data index))
                        (predictions (sl.tp:predictions leaf)))
                   (incf (sl.data:mref sums index 0) (aref predictions 0))
                   (incf (sl.data:mref sums index 1) (aref predictions 1))))
               (sl.tp:indexes state))
      (incf (sl.tp:contributions-count state))
      state)))


(defmethod sl.tp:contribute-predictions* ((parameters causal-classificaton)
                                          model
                                          data
                                          state
                                          parallel)
  (let* ((data-points-count (sl.data:data-points-count data))
         (number-of-classes (number-of-classes parameters)))
    (when (null state)
      (setf state (make-instance 'sl.tp:contributed-predictions
                                 :training-parameters parameters
                                 :sums (sl.data:make-data-matrix data-points-count
                                                                 (* 2 number-of-classes))
                                 :indexes (sl.data:iota-vector data-points-count))))
    (let ((sums (sl.tp:sums state))
          (root (sl.tp:root model)))
      (funcall (if parallel #'lparallel:pmap #'map)
               (lambda (index)
                 (let* ((leaf (sl.tp:leaf-for root data index))
                        (predictions (sl.tp:predictions leaf)))
                   (iterate
                     (for treatment from 0 to 1)
                     (iterate
                       (for class from 0 below number-of-classes)
                       (incf (sl.data:mref sums index (+ (* 2 class) treatment))
                             (sl.data:mref predictions treatment class))))))
               (sl.tp:indexes state))
      (incf (sl.tp:contributions-count state))
      state)))


(defmethod sl.tp:extract-predictions* ((parameters causal) state)
  (let ((predictions (sl.tp:sums state))
        (contributions-count (sl.tp:contributions-count state)))
    (sl.data:map-data-matrix (lambda (x) (/ x contributions-count))
                             predictions)))


(defmethod sl.mp:make-training-state ((parameters causal)
                                      &rest initargs)
  (apply #'make 'causal-tree-training-state initargs))


(defmethod initialize-instance :after ((state causal-tree-training-state)
                                       &rest initargs &key &allow-other-keys)
  (declare (ignore initargs))
  (sl.data:check-data-points (sl.mp:train-data state)
                             (sl.mp:target-data state)
                             (treatment state))
  (unless (= 1 (sl.data:attributes-count (treatment state)))
    (error 'cl-ds:invalid-argument-value
           :value (treatment state)
           :argument :treatment
           :format-arguments ":TREATMENT must have exactly 1 attribute.")))
