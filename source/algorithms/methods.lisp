(cl:in-package #:cl-grf.algorithms)


(defmethod calculate-score ((training-parameters single-impurity-classification)
                            split-array
                            target-data)
  (split-impurity training-parameters split-array target-data))


(defmethod cl-grf.tp:split* :around
    ((training-parameters scored-classification)
     training-state
     leaf)
  (declare (optimize (speed 3)))
  (when (<= (score leaf)
            (~> training-parameters minimal-difference))
    (return-from cl-grf.tp:split* nil))
  (call-next-method))


(defmethod cl-grf.tp:split*
    ((training-parameters scored-classification)
     training-state
     leaf)
  (declare (optimize (speed 3)))
  (bind ((training-data (cl-grf.tp:training-data training-state))
         (trials-count (cl-grf.tp:trials-count training-parameters))
         (minimal-difference (minimal-difference training-parameters))
         (score (score leaf))
         (minimal-size (cl-grf.tp:minimal-size training-parameters))
         (parallel (cl-grf.tp:parallel training-parameters))
         (target-data (cl-grf.tp:target-data training-state)))
    (declare (type fixnum trials-count)
             (type double-float score minimal-difference)
             (type boolean parallel)
             (type (simple-array double-float (* *))
                   training-data
                   target-data))
    (iterate
      (declare (type fixnum attempt left-length right-length
                     optimal-left-length optimal-right-length
                     optimal-attribute data-size)
               (type double-float
                     left-score right-score
                     minimal-score optimal-threshold))
      (with optimal-left-length = -1)
      (with optimal-right-length = -1)
      (with optimal-attribute = -1)
      (with minimal-score = most-positive-double-float)
      (with minimal-left-score = most-positive-double-float)
      (with minimal-right-score = most-positive-double-float)
      (with optimal-threshold = most-positive-double-float)
      (with data-size = (cl-grf.data:data-points-count training-data))
      (with split-array = (make-array data-size :element-type 'boolean
                                                :initial-element nil))
      (with optimal-array = (make-array data-size :element-type 'boolean
                                                  :initial-element nil))
      (for attempt from 0 below trials-count)
      (for (values attribute threshold) = (random-test training-data))
      (for (values left-length right-length) = (fill-split-array
                                                training-data attribute
                                                threshold split-array))
      (for (values left-score right-score) = (calculate-score
                                              training-parameters
                                              split-array
                                              target-data))
      (for split-score = (+ (* (/ left-length data-size) left-score)
                            (* (/ right-length data-size) right-score)))
      (when (and (< split-score minimal-score)
                 (>= left-length minimal-size)
                 (>= right-length minimal-size))
        (setf minimal-score split-score
              optimal-threshold threshold
              optimal-attribute attribute
              optimal-left-length left-length
              optimal-right-length right-length
              minimal-left-score left-score
              minimal-right-score right-score)
        (rotatef split-array optimal-array))
      (finally
       (let ((difference (- (the double-float score)
                            (the double-float minimal-score))))
         (declare (type double-float difference))
         (when (< difference minimal-difference)
           (return nil))
         (return (make 'scored-tree-node
                       :left-node (make-simple-node
                                   optimal-array
                                   minimal-left-score
                                   optimal-left-length
                                   nil
                                   parallel
                                   training-state
                                   optimal-attribute)
                       :right-node (make-simple-node
                                    optimal-array
                                    minimal-right-score
                                    optimal-right-length
                                    t
                                    nil
                                    training-state
                                    optimal-attribute)
                       :support data-size
                       :score score
                       :attribute (~> training-state
                                      cl-grf.tp:attribute-indexes
                                      (aref optimal-attribute))
                       :attribute-value optimal-threshold)))))))



(defmethod cl-grf.tp:make-leaf* ((training-parameters impurity-classification)
                                 training-state)
  (declare (optimize (speed 3)))
  (let* ((target-data (cl-grf.tp:target-data training-state))
         (number-of-classes (number-of-classes training-parameters))
         (data-points-count (cl-grf.data:data-points-count target-data))
         (predictions (cl-grf.data:make-data-matrix 1 number-of-classes)))
    (declare (type fixnum number-of-classes data-points-count)
             (type cl-grf.data:data-matrix target-data predictions))
    (iterate
      (declare (type fixnum i index))
      (for i from 0 below data-points-count)
      (for index = (truncate (cl-grf.data:mref target-data i 0)))
      (incf (cl-grf.data:mref predictions 0 index)))
    (make-instance 'scored-leaf-node
                   :support (cl-grf.data:data-points-count target-data)
                   :predictions predictions
                   :score (total-impurity training-parameters
                                          target-data))))


(defmethod cl-grf.mp:make-model ((parameters impurity-classification)
                                 train-data
                                 target-data
                                 &optional weights)
  (declare (ignore weights))
  (let* ((attributes (iterate
                       (with attributes-count =
                             (cl-grf.data:attributes-count train-data))
                       (with result = (make-array attributes-count
                                                  :element-type 'fixnum
                                                  :initial-element 0))
                       (for i from 0 below attributes-count)
                       (setf (aref result i) i)
                       (finally (return result))))
         (state (make 'cl-grf.tp:fundamental-training-state
                      :training-parameters parameters
                      :attribute-indexes attributes
                      :target-data target-data
                      :training-data train-data))
         (leaf (cl-grf.tp:make-leaf state))
         (tree (cl-grf.tp:split state leaf)))
    (if (null tree)
        leaf
        tree)))


(defmethod shared-initialize :after ((parameters impurity-classification)
                                     slot-names
                                     &rest initargs)
  (declare (ignore slot-names initargs))
  (let ((minimal-difference (minimal-difference parameters)))
    (unless (typep minimal-difference 'double-float)
      (error 'type-error
             :expected-type 'double-float
             :datum minimal-difference))
    (when (< minimal-difference 0.0d0)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :minimal-difference
             :value minimal-difference
             :format-control "Minimal difference in the impurity-classification must not be negative."))))


(defmethod shared-initialize :after ((parameters single-impurity-classification)
                                     slot-names
                                     &rest initargs)
  (declare (ignore slot-names initargs))
  (let ((number-of-classes (number-of-classes parameters)))
    ;; TODO should also check class weights
    (unless (integerp number-of-classes)
      (error 'type-error
             :expected-type 'integer
             :datum number-of-classes))
    (when (< number-of-classes 2)
      (error 'cl-ds:argument-value-out-of-bounds
             :bounds '(>= :number-of-classes 2)
             :value number-of-classes
             :argument :number-of-classes
             :format-control "Classification requires at least 2 classes for classification."))))


(defmethod cl-grf.performance:performance-metric
    ((parameters single-impurity-classification)
     target
     predictions)
  (bind ((number-of-classes (the fixnum (number-of-classes parameters)))
         (data-points-count (cl-grf.data:data-points-count target))
         ((:flet prediction (prediction))
          (declare (optimize (speed 3) (safety 0)))
          (cl-grf.data:check-data-points prediction)
          (iterate
            (declare (type fixnum i))
            (for i from 0 below number-of-classes)
            (finding i maximizing (cl-grf.data:mref prediction 0 i))))
         (result (cl-grf.performance:make-confusion-matrix number-of-classes)))
    (iterate
      (declare (type fixnum i)
               (optimize (speed 3)))
      (for i from 0 below data-points-count)
      (for expected = (truncate (cl-grf.data:mref target i 0)))
      (for predicted = (prediction (aref predictions i)))
      (incf (cl-grf.performance:at-confusion-matrix
             result expected predicted)))
    result))


(defmethod cl-grf.performance:average-performance-metric ((parameters impurity-classification)
                                                          metrics)
  (iterate
    (with result = (~> parameters number-of-classes
                       cl-grf.performance:make-confusion-matrix))
    (for i from 0 below (length metrics))
    (for confusion-matrix = (aref metrics i))
    (sum-matrices confusion-matrix result)
    (finally (return result))))
