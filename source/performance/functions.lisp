(cl:in-package #:cl-grf.performance)


(defun cross-validation (model-parameters number-of-folds
                         train-data target-data &optional parallel)
  (cl-grf.data:check-data-points train-data target-data)
  (~> train-data
      cl-grf.data:data-points-count
      (cl-grf.data:cross-validation-folds number-of-folds)
      (cl-ds.alg:on-each
       (lambda (train.test)
         (bind (((train . test) train.test)
                (model (cl-grf.mp:make-model
                        model-parameters
                        (cl-grf.data:sample train-data
                                            :data-points train)
                        (cl-grf.data:sample target-data
                                            :data-points train)))
                (test-target-data (cl-grf.data:sample target-data
                                                      :data-points test))
                (test-train-data (cl-grf.data:sample train-data
                                                     :data-points test))
                (test-predictions (cl-grf.mp:predict model
                                                     test-train-data
                                                     parallel)))
           (performance-metric model-parameters
                               test-target-data
                               test-predictions))))
      cl-ds.alg:to-vector
      (average-performance-metric model-parameters _)))


(defun make-confusion-matrix (number-of-classes)
  (make-array (list number-of-classes number-of-classes)
              :element-type 'fixnum))


(defun number-of-classes (confusion-matrix)
  (array-dimension confusion-matrix 0))


(defun at-confusion-matrix (confusion-matrix expected-class predicted-class)
  (aref confusion-matrix expected-class predicted-class))


(defun (setf at-confusion-matrix) (new-value confusion-matrix
                                   expected-class predicted-class)
  (setf (aref confusion-matrix expected-class predicted-class)
        new-value))


(defun total (confusion-matrix)
  (iterate
    (for i from 0 below (array-total-size confusion-matrix))
    (sum (row-major-aref confusion-matrix i))))


(defun two-class-confusion-matrix-from-general-confusion-matrix
    (confusion-matrix class &optional (result (make-confusion-matrix 2)))
  (iterate
    (with number-of-classes = (array-dimension confusion-matrix 0))
    (for expected from 0 below number-of-classes)
    (for true/false = (if (= expected class) 1 0))
    (iterate
      (for predicted from 0 below number-of-classes)
      (for positive/negative = (if (= predicted class) 1 0))
      (incf (at-confusion-matrix result
                                 true/false
                                 positive/negative)
            (at-confusion-matrix confusion-matrix expected predicted)))
    (finally (return result))))


(defun fold-general-confusion-matrix (confusion-matrix)
  (iterate
    (with result = (make-confusion-matrix 2))
    (for i from 0 below (number-of-classes confusion-matrix))
    (two-class-confusion-matrix-from-general-confusion-matrix
     confusion-matrix
     i
     result)
    (finally (return result))))


(defun accuracy (confusion-matrix)
  (coerce (/ (iterate
               (for i from 0 below (array-dimension confusion-matrix 0))
               (sum (at-confusion-matrix confusion-matrix i i)))
             (total confusion-matrix))
          'double-float))


(defun recall (confusion-matrix)
  (let ((folded (fold-general-confusion-matrix confusion-matrix)))
    (coerce (/ (aref folded 1 1)
               (+ (aref folded 0 1)
                  (aref folded 1 1)))
            'double-float)))


(defun specificity (confusion-matrix)
  (let ((folded (fold-general-confusion-matrix confusion-matrix)))
    (coerce (/ (aref folded 0 0)
               (+ (aref folded 0 0)
                  (aref folded 0 1)))
            'double-float)))


(defun precision (confusion-matrix)
  (let ((folded (fold-general-confusion-matrix confusion-matrix)))
    (coerce (/ (aref folded 1 1)
               (+ (aref folded 1 0)
                  (aref folded 1 1)))
            'double-float)))


(defun f1-score (confusion-matrix)
  (coerce (/ 2 (+ (/ 1 (precision confusion-matrix))
                  (/ 1 (recall confusion-matrix))))
          'double-float))
