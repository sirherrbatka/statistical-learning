(cl:in-package #:cl-grf.mp)


(defun cross-validation (model-parameters number-of-folds
                         train-data target-data)
  (cl-grf.data:check-data-points train-data target-data)
  (~> train-data
      cl-grf.data:data-points-count
      (cl-grf.data:cross-validation-folds number-of-folds)
      (cl-ds.alg:on-each
       (lambda (train.test)
         (bind (((train . test) train.test)
                (model (make-model model-parameters
                                   (cl-grf.data:sample train-data
                                                       :data-points train)
                                   (cl-grf.data:sample target-data
                                                       :data-points train)))
                (test-target-data (cl-grf.data:sample train-data
                                                      :data-points test))
                (test-predictions (predict model test-target-data)))
           (performance-metric model-parameters
                               test-predictions
                               test-target-data))))
      cl-ds.alg:to-vector
      (average-performance-metric model-parameters _)))


(defun false-positive (confusion-matrix)
  (- (positive confusion-matrix)
     (true-positive confusion-matrix)))


(defun false-negative (confusion-matrix)
  (- (negative confusion-matrix)
     (true-negative confusion-matrix)))


(defun true (confusion-matrix)
  (+ (true-positive confusion-matrix)
     (true-negative confusion-matrix)))


(defun false (confusion-matrix)
  (+ (false-negative confusion-matrix)
     (false-positive confusion-matrix)))


(defun total (confusion-matrix)
  (+ (positive confusion-matrix)
     (negative confusion-matrix)))


(defun precision (confusion-matrix)
  (coerce (/ (true-positive confusion-matrix)
             (positive confusion-matrix))
          'double-float))


(defun recall (confusion-matrix)
  (coerce (/ (true-positive confusion-matrix)
             (+ (true-positive confusion-matrix)
                (false-positive confusion-matrix)))
          'double-float))


(defun specificity (confusion-matrix)
  (coerce (/ (true-negative confusion-matrix)
             (negative confusion-matrix))
          'double-float))


(defun accuracy (confusion-matrix)
  (coerce (/ (total-true confusion-matrix)
             (total confusion-matrix))
          'double-float))


(defun f1-score (confusion-matrix)
  (coerce (/ 2 (+ (/ 1 (precision confusion-matrix))
                  (/ 1 (recall confusion-matrix))))
          'double-float))
