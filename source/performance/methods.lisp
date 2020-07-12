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
  (bind ((precision 1000)
         (counters (make-array `(,precision 2)
                               :initial-element 0))
         (total (sl.data:data-points-count target))
         (positive-count 0)
         (negative-count 0))
    (iterate
      (for i from 0 below total)
      (for expected = (truncate (sl.data:mref target i 0)))
      (for truep = (not (zerop expected)))
      (for probability = (sl.data:mref predictions i 1))
      (if truep
          (incf positive-count)
          (incf negative-count))
      (iterate
        (for j from 0 below precision)
        (for threshold from (/ 1 precision))
        (cl-ds.utils:cond+ (truep (> probability threshold))
          ((t t) (incf (aref counters j 1)))
          ((nil t) (incf (aref counters j 0)))
          ((nil nil) nil)
          ((t nil) nil))))
    (iterate
      (for i from 0 below precision)
      (for true-positive = (aref counters i 1))
      (for false-positive = (aref counters i 0))
      (for tpr = (/ true-positive positive-count))
      (for fpr = (/ false-positive negative-count))
      (setf (aref counters i 1) tpr)
      (setf (aref counters i 0) fpr)
      (for p-tpr previous tpr initially 0)
      (for p-fpr previous fpr initially 0)
      (for fpr-span = (- fpr p-fpr))
      (for tpr-span = (- tpr p-tpr))
      (for field = (+ (* fpr-span p-tpr)
                      (/ (* tpr-span fpr-span)
                         2)))
      (sum field into result)
      (return (values result counters)))))


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
