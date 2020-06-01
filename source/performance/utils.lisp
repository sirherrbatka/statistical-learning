(cl:in-package #:cl-grf.performance)


(defun permutate-attribute (data attribute indexes)
  (cl-grf.data:bind-data-matrix-dimensions ((data-points attributes data))
    (ensure indexes (cl-grf.data:iota-vector data-points))
    (cl-grf.data:reshuffle indexes)
    (iterate
      (with result = (copy-array data))
      (for i from 0 below data-points)
      (setf (cl-grf.data:mref result i attribute)
            (cl-grf.data:mref data (aref indexes i) attribute))
      (finally (return result)))))


(defun calculate-features-importance-from-permutations
    (model model-parameters errors
     test-train-data test-target-data parallel)
  (cl-grf.data:check-data-points test-train-data test-target-data)
  (let* ((attributes-count (cl-grf.data:attributes-count test-train-data))
         (error-differences (make-array (array-dimensions errors)
                                        :element-type 'double-float))
         (result (make-array attributes-count
                             :initial-element 0.0d0
                             :element-type 'double-float)))
    (iterate
      (with indexes = (~> test-train-data
                         cl-grf.data:data-points-count
                         cl-grf.data:iota-vector))
      (for i from 0 below attributes-count)
      (for permutated = (permutate-attribute test-train-data i indexes))
      (for predictions = (cl-grf.mp:predict model permutated parallel))
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
