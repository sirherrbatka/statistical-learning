(cl:in-package #:statistical-learning.performance)


(defun permutate-attribute (data attribute indexes)
  (statistical-learning.data:bind-data-matrix-dimensions ((data-points attributes data))
    (ensure indexes (statistical-learning.data:iota-vector data-points))
    (statistical-learning.data:reshuffle indexes)
    (iterate
      (with result = (copy-array data))
      (for i from 0 below data-points)
      (setf (statistical-learning.data:mref result i attribute)
            (statistical-learning.data:mref data (aref indexes i) attribute))
      (finally (return result)))))


(defun calculate-features-importance-from-permutations
    (model model-parameters errors
     test-train-data test-target-data parallel)
  (statistical-learning.data:check-data-points test-train-data test-target-data)
  (let* ((attributes-count (statistical-learning.data:attributes-count test-train-data))
         (error-differences (make-array (array-dimensions errors)
                                        :element-type 'double-float))
         (result (make-array attributes-count
                             :initial-element 0.0d0
                             :element-type 'double-float)))
    (iterate
      (with indexes = (~> test-train-data
                          statistical-learning.data:data-points-count
                          statistical-learning.data:iota-vector))
      (for i from 0 below attributes-count)
      (for permutated = (permutate-attribute test-train-data i indexes))
      (for predictions = (statistical-learning.mp:predict model permutated parallel))
      (for permutated-errors = (errors model-parameters test-target-data predictions))
      (map-into error-differences
                (lambda (a b) (max 0.0d0 (- a b)))
                permutated-errors
                errors)
      (for mean-change = (mean error-differences))
      (for sd = (alexandria:standard-deviation error-differences :biased nil))
      (for feature-importance = (if (zerop sd) 0.0d0 (/ mean-change sd)))
      (setf (aref result i) feature-importance)
      (finally (return result)))))


(defun sum-matrices (matrix result)
  (iterate
    (for i from 0 below (array-total-size matrix))
    (incf (row-major-aref result i)
          (row-major-aref matrix i)))
  result)
