(cl:in-package #:cl-grf.algorithms)


(defmethod calculate-score ((training-parameters single-impurity-classification)
                            split-array
                            state)
  (split-impurity training-parameters
                  split-array
                  (cl-grf.tp:target-data state)))


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
    ((training-parameters scored-training)
     training-state
     leaf)
  (declare (optimize (speed 3) (safety 0)))
  (bind ((training-data (cl-grf.tp:training-data training-state))
         (trials-count (cl-grf.tp:trials-count training-parameters))
         (minimal-difference (minimal-difference training-parameters))
         (score (score leaf))
         (minimal-size (cl-grf.tp:minimal-size training-parameters))
         (parallel (cl-grf.tp:parallel training-parameters))
         (attributes (cl-grf.tp:attribute-indexes training-state)))
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
      (for (values attribute threshold) = (random-test attributes training-data))
      (for (values left-length right-length) = (fill-split-array
                                                training-data
                                                (aref attributes attribute)
                                                threshold split-array))
      (when (or (< left-length minimal-size)
                (< right-length minimal-size))
        (next-iteration))
      (for (values left-score right-score) = (calculate-score
                                              training-parameters
                                              split-array
                                              training-state))
      (for split-score = (+ (* (/ left-length data-size) left-score)
                            (* (/ right-length data-size) right-score)))
      (when (< split-score minimal-score)
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
         (let ((new-attributes (subsample-vector attributes
                                                 optimal-attribute)))
           (return (make 'scored-tree-node
                         :left-node (make-simple-node
                                     optimal-array
                                     minimal-left-score
                                     optimal-left-length
                                     nil
                                     parallel
                                     training-state
                                     new-attributes)
                         :right-node (make-simple-node
                                      optimal-array
                                      minimal-right-score
                                      optimal-right-length
                                      t
                                      nil
                                      training-state
                                      new-attributes)
                         :support data-size
                         :score score
                         :attribute (aref attributes optimal-attribute)
                         :attribute-value optimal-threshold))))))))



(defmethod cl-grf.tp:make-leaf* ((training-parameters single-impurity-classification)
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


(defmethod cl-grf.tp:make-leaf* ((training-parameters regression)
                                 training-state)
  (declare (optimize (speed 3)))
  (let* ((target-data (cl-grf.tp:target-data training-state))
         (data-points-count (cl-grf.data:data-points-count target-data)))
    (declare (type fixnum data-points-count))
    (iterate
      (declare (type fixnum i)
               (type double-float sum))
      (with sum = 0.0d0)
      (for i from 0 below data-points-count)
      (incf sum (cl-grf.data:mref target-data i 0))
      (finally (return (make-instance
                        'scored-leaf-node
                        :support (cl-grf.data:data-points-count target-data)
                        :predictions (/ sum data-points-count)
                        :score (~> data-points-count
                                   (make-array :initial-element nil
                                               :element-type 'boolean)
                                   (calculate-score training-parameters
                                                    _
                                                    training-state))))))))


(defmethod cl-grf.mp:make-model ((parameters gradient-boost-regression)
                                 train-data
                                 target-data
                                 &key attributes
                                   predictions
                                   expected-value
                                 &allow-other-keys)
  (let* ((differences
           (if (null predictions)
               (cl-grf.data:map-data-matrix (lambda (x)
                                              (- x expected-value))
                                            target-data)
               (iterate
                 (with result = (cl-grf.data:make-data-matrix-like target-data))
                 (for i from 0 below (cl-grf.data:data-points-count result))
                 (setf (cl-grf.data:mref result i 0)
                       (- (cl-grf.data:mref target-data i 0)
                          (cl-grf.data:mref predictions i 0)))
                 (finally (return result)))))
         (state (make 'cl-grf.tp:fundamental-training-state
                      :training-parameters parameters
                      :attribute-indexes attributes
                      :target-data differences
                      :training-data train-data))
         (leaf (cl-grf.tp:make-leaf state))
         (tree (cl-grf.tp:split state leaf)))
    (make 'gradient-boost-model
          :parameters parameters
          :expected-value expected-value
          :root (if (null tree) leaf tree))))


(defmethod cl-grf.mp:make-model ((parameters scored-training)
                                 train-data
                                 target-data
                                 &key attributes &allow-other-keys)
  (let* ((state (make 'cl-grf.tp:fundamental-training-state
                      :training-parameters parameters
                      :attribute-indexes attributes
                      :target-data target-data
                      :training-data train-data))
         (leaf (cl-grf.tp:make-leaf state))
         (tree (cl-grf.tp:split state leaf)))
    (make 'cl-grf.tp:tree-model
          :parameters parameters
          :root (if (null tree) leaf tree))))


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
  (cl-grf.data:check-data-points target predictions)
  (bind ((number-of-classes (the fixnum (number-of-classes parameters)))
         (data-points-count (cl-grf.data:data-points-count target))
         ((:flet prediction (prediction))
          (declare (optimize (speed 3) (safety 0)))
          (iterate
            (declare (type fixnum i))
            (for i from 0 below number-of-classes)
            (finding i maximizing (cl-grf.data:mref predictions prediction i))))
         (result (cl-grf.performance:make-confusion-matrix number-of-classes)))
    (iterate
      (declare (type fixnum i)
               (optimize (speed 3)))
      (for i from 0 below data-points-count)
      (for expected = (truncate (cl-grf.data:mref target i 0)))
      (for predicted = (prediction i))
      (incf (cl-grf.performance:at-confusion-matrix
             result expected predicted)))
    result))


(defmethod cl-grf.performance:average-performance-metric
    ((parameters classification)
     metrics)
  (iterate
    (with result = (~> parameters number-of-classes
                       cl-grf.performance:make-confusion-matrix))
    (for i from 0 below (length metrics))
    (for confusion-matrix = (aref metrics i))
    (sum-matrices confusion-matrix result)
    (finally (return result))))


(defmethod cl-grf.performance:errors ((parameters classification)
                                      target
                                      predictions)
  (declare (optimize (speed 3) (safety 0))
           (type simple-vector predictions)
           (type cl-grf.data:data-matrix target))
  (let* ((data-points-count (cl-grf.data:data-points-count target))
         (result (make-array data-points-count :element-type 'double-float)))
    (declare (type (simple-array double-float (*)) result))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below data-points-count)
      (for expected = (truncate (cl-grf.data:mref target i 0)))
      (setf (aref result i) (- 1.0d0 (cl-grf.data:mref predictions
                                                       i
                                                       expected))))
    result))


(defmethod cl-grf.performance:errors ((parameters regression)
                                      target
                                      predictions)
  (iterate
    (with result = (make-array (cl-grf.data:data-points-count predictions)
                               :element-type 'double-float
                               :initial-element 0.0d0))
    (for i from 0 below (length predictions))
    (for er = (- (cl-grf.data:mref target i 0)
                 (cl-grf.data:mref predictions i 0)))
    (setf (aref result i) (* er er))
    (finally (return result))))


(defmethod cl-grf.performance:average-performance-metric ((parameters regression)
                                                          metrics)
  (mean metrics))


(defclass gathered-predictions ()
  ((%contributions-count :initarg :contributions-count
                         :accessor contributions-count)
   (%indexes :initarg :indexes
             :reader indexes)
   (%sums :initarg :sums
          :reader sums))
  (:default-initargs :contributions-count 0))


(defclass gradient-boost-gathered-predictions (gathered-predictions)
  ((%expected-value :initarg :expected-value
                    :reader expected-value)))


(defmethod cl-grf.tp:contribute-predictions ((parameters classification)
                                             model
                                             data
                                             state
                                             parallel)
  (cl-grf.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (let ((number-of-classes (number-of-classes parameters)))
      (when (null state)
        (setf state (make 'gathered-predictions
                          :indexes (cl-grf.data:iota-vector data-points-count)
                          :sums (cl-grf.data:make-data-matrix data-points-count
                                                              number-of-classes))))
      (let* ((sums (sums state))
             (root (cl-grf.tp:root model)))
        (funcall (if parallel #'lparallel:pmap #'map)
                 nil
                 (lambda (data-point)
                   (iterate
                     (declare (type fixnum j))
                     (with leaf = (cl-grf.tp:leaf-for root data data-point))
                     (with predictions = (predictions leaf))
                     (with support = (support leaf))
                     (for j from 0 below number-of-classes)
                     (for class-support = (cl-grf.data:mref predictions 0 j))
                     (incf (cl-grf.data:mref sums data-point j)
                           (/ class-support support))))
                 (indexes state))))
    (incf (contributions-count state))
    state))


(defmethod cl-grf.tp:contribute-predictions ((parameters basic-regression)
                                             model
                                             data
                                             state
                                             parallel)
  (cl-grf.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (when (null state)
      (setf state (make 'gathered-predictions
                        :indexes (cl-grf.data:iota-vector data-points-count)
                        :sums (cl-grf.data:make-data-matrix data-points-count
                                                            1))))
    (let* ((sums (sums state))
           (root (cl-grf.tp:root model)))
      (funcall (if parallel #'lparallel:pmap #'map)
               nil
               (lambda (data-point)
                 (let* ((leaf (cl-grf.tp:leaf-for root data data-point))
                        (predictions (predictions leaf)))
                   (incf (cl-grf.data:mref sums data-point 0)
                         predictions)))
               (indexes state)))
    (incf (contributions-count state))
    state))


(defmethod cl-grf.tp:contribute-predictions ((parameters gradient-boost-regression)
                                             model
                                             data
                                             state
                                             parallel)
  (cl-grf.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (when (null state)
      (setf state (make 'gradient-boost-gathered-predictions
                        :indexes (cl-grf.data:iota-vector data-points-count)
                        :expected-value (expected-value model)
                        :sums (cl-grf.data:make-data-matrix data-points-count
                                                            1))))
    (let* ((sums (sums state))
           (learning-rate (learning-rate parameters))
           (root (cl-grf.tp:root model)))
      (funcall (if parallel #'lparallel:pmap #'map)
               nil
               (lambda (data-point)
                 (let* ((leaf (cl-grf.tp:leaf-for root data data-point))
                        (predictions (predictions leaf)))
                   (incf (cl-grf.data:mref sums data-point 0)
                         (* learning-rate predictions))))
               (indexes state)))
    (incf (contributions-count state))
    state))


(defmethod cl-grf.tp:extract-predictions ((state gathered-predictions))
  (let ((count (contributions-count state)))
    (cl-grf.data:map-data-matrix (lambda (value) (/ value count))
                                 (sums state))))


(defmethod cl-grf.tp:extract-predictions ((state gradient-boost-gathered-predictions))
  (let* ((count (contributions-count state))
         (sums (sums state))
         (expected-value (expected-value state))
         (result (cl-grf.data:make-data-matrix-like sums)))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below (cl-grf.data:data-points-count sums))
      (setf (cl-grf.data:mref result i 0)
            (+ expected-value
               (cl-grf.data:mref sums i 0))))
    result))


(defun regression-score (split-array target-data)
  (let ((left-sum 0.0d0)
        (right-sum 0.0d0)
        (left-count 0)
        (right-count 0))
    (declare (type double-float left-sum right-sum)
             (type cl-grf.data:data-matrix target-data)
             (type fixnum left-count right-count))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below (length split-array))
      (for right-p = (aref split-array i))
      (for value = (cl-grf.data:mref target-data i 0))
      (if right-p
          (setf right-count (1+ right-count)
                right-sum (+ right-sum value))
          (setf left-count (1+ left-count)
                left-sum (+ left-sum value))))
    (iterate
      (declare (type double-float
                     left-error right-error
                     left-avg right-avg)
               (type fixnum i))
      (with left-error = 0.0d0)
      (with right-error = 0.0d0)
      (with left-avg = (if (zerop left-count)
                           0.0d0
                           (/ left-sum left-count)))
      (with right-avg = (if (zerop right-count)
                            0.0d0
                            (/ right-sum right-count)))
      (for i from 0 below (length split-array))
      (for right-p = (aref split-array i))
      (for value = (cl-grf.data:mref target-data i 0))
      (if right-p
          (incf right-error (square (- value right-avg)))
          (incf left-error (square (- value left-avg))))
      (finally (return (values (if (zerop left-count)
                                   0.0d0
                                   (/ left-error left-count))
                               (if (zerop right-count)
                                   0.0d0
                                   (/ right-error right-count))))))))


(defmethod calculate-score ((training-parameters regression)
                            split-array
                            training-state)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array boolean (*)) split-array))
  (~>> training-state
       cl-grf.tp:target-data
       (regression-score split-array)))
