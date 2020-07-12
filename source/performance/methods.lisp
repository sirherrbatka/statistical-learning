(cl:in-package #:statistical-learning.performance)


(defmethod performance-metric* :before ((parameters sl.mp:fundamental-model-parameters)
                                        type target predictions weights)
  (check-type weights (or null sl.data:double-float-data-matrix))
  (statistical-learning.data:check-data-points target predictions))


(defmethod performance-metric* ((parameters regression)
                                (type (eql :mean-squared-error))
                                target
                                predictions
                                weights)
  (iterate
    (with sum = 0.0d0)
    (with count = (sl.data:data-points-count predictions))
    (for i from 0 below count)
    (for er = (- (sl.data:mref target i 0)
                 (sl.data:mref predictions i 0)))
    (incf sum (* (if (null weights) 1.0d0 (sl.data:mref weights i 0))
                 (* er er)))
    (finally (return (/ sum count)))))


(defmethod performance-metric* ((parameters sl.mp:fundamental-model-parameters)
                                (type (eql :default))
                                target
                                predictions
                                weights)
  (performance-metric* parameters
                       (default-performance-metric parameters)
                       target predictions weights))


(defmethod default-performance-metric ((parameters classification))
  :confusion-matrix)


(defmethod default-performance-metric ((parameters regression))
  :mean-squared-error)


(defmethod errors ((parameters regression) target predictions)
  (iterate
    (with result = (make-array (sl.data:data-points-count predictions)
                               :element-type 'double-float
                               :initial-element 0.0d0))
    (for i from 0 below (sl.data:data-points-count predictions))
    (for er = (- (sl.data:mref target i 0)
                 (sl.data:mref predictions i 0)))
    (setf (aref result i) (* er er))
    (finally (return result))))


(defmethod average-performance-metric* ((parameters regression)
                                        (type (eql :mean-squared-error))
                                        metrics)
  (mean metrics))


(defmethod average-performance-metric* ((parameters classification)
                                        (type (eql :roc-auc))
                                        metrics)
  (mean metrics))


(defmethod errors ((parameters classification)
                   target
                   predictions)
  (declare (optimize (speed 3) (safety 0))
           (type sl.data:double-float-data-matrix target predictions))
  (let* ((data-points-count (sl.data:data-points-count target))
         (result (make-array data-points-count :element-type 'double-float)))
    (declare (type (simple-array double-float (*)) result))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below data-points-count)
      (for expected = (truncate (sl.data:mref target i 0)))
      (setf (aref result i) (- 1.0d0 (sl.data:mref predictions
                                                   i
                                                   expected))))
    result))


(defmethod average-performance-metric*
    ((parameters classification)
     (type (eql :confusion-matrix))
     metrics)
  (iterate
    (with result = (~> metrics first-elt copy-array))
    (for i from 1 below (length metrics))
    (for confusion-matrix = (aref metrics i))
    (sum-matrices confusion-matrix result)
    (finally (return result))))


(defmethod performance-metric*
    ((parameters classification)
     (type (eql :confusion-matrix))
     target
     predictions
     weights)
  (sl.data:check-data-points target predictions)
  (bind ((number-of-classes (the fixnum (sl.opt:number-of-classes parameters)))
         (data-points-count (sl.data:data-points-count target))
         ((:flet prediction (prediction))
          (declare (optimize (speed 3) (safety 0)))
          (iterate
            (declare (type fixnum i))
            (for i from 0 below number-of-classes)
            (finding i maximizing (sl.data:mref predictions prediction i))))
         (result (make-confusion-matrix number-of-classes)))
    (iterate
      (declare (type fixnum i)
               (optimize (speed 3)))
      (for i from 0 below data-points-count)
      (for expected = (truncate (sl.data:mref target i 0)))
      (for predicted = (prediction i))
      (incf (sl.perf:at-confusion-matrix result expected predicted)
            (if (null weights)
                1.0d0
                (sl.data:mref weights i 0))))
    result))


(defmethod performance-metric*
    ((parameters classification)
     (type (eql :roc-auc))
     target
     predictions
     weights)
  (bind ((precision 5000)
         (counters (map-into (make-array precision)
                             (curry #'list 0 0)))
         (positive 0)
         (negative 0)
         (total (sl.data:data-points-count target))
         ((:flet calculate-rate (stats))
          (bind (((true-positive false-positive) stats))
            (vector (if (zerop negative)
                        0
                        (/ false-positive negative))
                    (if (zerop positive)
                        0
                        (/ true-positive positive))))))
    (iterate
      (with step = (/ 1 precision))
      (for i from 0 below total)
      (for expected = (sl.data:mref target i 0))
      (for truep = (= 1 expected))
      (for probability = (sl.data:mref predictions i 1))
      (if truep
          (incf positive)
          (incf negative))
      (iterate
        (for j from 0 below precision)
        (for threshold from step by step)
        (for counter = (aref counters j))
        (for positive = (> probability threshold))
        (when positive
          (if truep
              (incf (first counter))
              (incf (second counter))))))
    (iterate
      (for (fpr tpr) in-vector
           (~> (cl-ds.alg:on-each counters #'calculate-rate)
               (cl-ds.alg:partition-if #'= :key #'first-elt)
               (cl-ds.alg:on-each (compose #'cl-ds.math:average
                                           #'cl-ds.alg:array-elementwise))
               (cl-ds.alg:to-vector :key (rcurry #'coerce 'list))
               nreverse))
      (for p-tpr previous tpr initially 0)
      (for p-fpr previous fpr initially 0)
      (for fpr-span = (- fpr p-fpr))
      (for tpr-span = (- tpr p-tpr))
      (for field = (+ (* fpr-span p-tpr)
                      (/ (* tpr-span fpr-span)
                         2)))
      (sum field into result)
      (finally (return (coerce result 'double-float))))))


(defmethod average-performance-metric ((parameters sl.mp:fundamental-model-parameters)
                                       metrics
                                       &key (type :default))
  (average-performance-metric* parameters type metrics))


(defmethod average-performance-metric* ((parameters sl.mp:fundamental-model-parameters)
                                        (type (eql :default))
                                        metrics)
  (average-performance-metric* parameters
                               (default-performance-metric parameters)
                               metrics))
