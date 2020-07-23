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
  (list (cl-ds.math:average metrics :key #'first)
        (apply #'append (map 'list #'second metrics))))


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
  (bind ((positive 0)
         (negative 0)
         (total (sl.data:data-points-count target))
         (points (iterate
                   (with result = (make-array total))
                   (for i from 0 below total)
                   (for truep = (= 1 (sl.data:mref target i 0)))
                   (if truep (incf positive) (incf negative))
                   (setf (aref result i)
                         (list (sl.data:mref predictions i 1)
                               truep))
                   (finally (return (sort result #'> :key #'first)))))
         (curve (vect (list 1.0d0 0.0d0 0.0d0)))
         (field 0.0d0)
         ((:flet update-field (threshold fpr tpr))
          (bind (((p-threshold p-fpr p-tpr) (last-elt curve))
                 (tpr-span (- tpr p-tpr))
                 (fpr-span (- fpr p-fpr)))
            (incf field (+ (* fpr-span p-tpr)
                           (/ (* tpr-span fpr-span)
                              2)))
            (unless (zerop fpr-span)
              (vector-push-extend (list threshold fpr tpr)
                                  curve)))))
    (iterate
      (with fp = 0)
      (with tp = 0)
      (for (score target) in-vector points)
      (for p-score previous score initially 1.0d0)
      (unless (= score p-score)
        (let ((fpr (coerce (/ fp negative)
                           'double-float))
              (tpr (coerce (/ tp positive)
                           'double-float)))
          (update-field p-score fpr tpr)))
      (if target (incf tp) (incf fp))
      (finally (update-field 0.0d0 1.0d0 1.0d0)))
    (iterate
      (with data-matrix = (sl.data:make-data-matrix (length curve)
                                                    3))
      (for i from 0 below (length curve))
      (for (threshold fpr tpr) = (aref curve i))
      (setf (sl.data:mref data-matrix i 0) threshold
            (sl.data:mref data-matrix i 1) fpr
            (sl.data:mref data-matrix i 2) tpr)
      (finally (return (list field (list data-matrix)))))))


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
