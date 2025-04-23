(cl:in-package #:statistical-learning.performance)


(defun no-after (model train target)
  (declare (ignore train target))
  model)


(defun cross-validation (model-parameters number-of-folds
                         train-data target-data
                         &key weights parallel performance-function average-performance-function
                           (performance-type :default)
                           (after #'no-after)
                         &allow-other-keys)
  (statistical-learning.data:check-data-points train-data target-data)
  (~> train-data
      statistical-learning.data:data-points-count
      (statistical-learning.data:cross-validation-folds number-of-folds)
      (cl-ds.alg:on-each
       (lambda (train.test)
         (bind (((train . test) train.test)
                ((:flet sampled-weights (sample))
                 (if (null weights)
                     nil
                     (map '(vector single-float)
                          (lambda (i) (aref weights i))
                          sample)))
                (fold-train-data (statistical-learning.data:sample
                                  train-data
                                  :data-points train))
                (fold-target-data (statistical-learning.data:sample
                                   target-data
                                   :data-points train))
                (model (~>> (statistical-learning.mp:make-supervised-model
                             model-parameters
                             fold-train-data
                             fold-target-data
                             :weights (sampled-weights train))
                            (funcall after _ fold-train-data fold-target-data)))
                (test-target-data (statistical-learning.data:sample (sl.data:wrap target-data)
                                                                    :data-points test))
                (test-train-data (statistical-learning.data:sample (sl.data:wrap train-data)
                                                                   :data-points test))
                (test-predictions (statistical-learning.mp:predict model
                                                                   test-train-data
                                                                   parallel)))
           (if (null performance-function)
               (performance-metric model-parameters
                                   test-target-data
                                   test-predictions
                                   :type performance-type
                                   :weights weights)
               (funcall performance-function
                        test-target-data
                        test-predictions
                        :weights weights)))))
      cl-ds.alg:to-vector
      (average-performance model-parameters
                           average-performance-function
                           _
                           performance-type)))


(defun attributes-importance* (model train-data target-data &key parallel weights)
  (let* ((predictions (statistical-learning.mp:predict model train-data parallel))
         (model-parameters (statistical-learning.mp:parameters model))
         (errors (errors model-parameters
                         target-data
                         predictions)))
    (calculate-features-importance-from-permutations model
                                                     model-parameters
                                                     errors
                                                     train-data
                                                     target-data
                                                     parallel
                                                     weights)))


(defun attributes-importance (model-parameters number-of-folds
                              train-data target-data &key parallel)
  (statistical-learning.data:check-data-points train-data target-data)
  (~> train-data
      statistical-learning.data:data-points-count
      (statistical-learning.data:cross-validation-folds number-of-folds)
      (cl-ds.alg:on-each
       (lambda (train.test)
         (bind (((train . test) train.test)
                (train-train-data (sl.data:sample train-data
                                                  :data-points train))
                (train-target-data (sl.data:sample target-data
                                                   :data-points train))
                (model (sl.mp:make-supervised-model model-parameters
                                                    train-train-data
                                                    train-target-data))
                (test-target-data (sl.data:sample target-data
                                                  :data-points test))
                (test-train-data (sl.data:sample train-data
                                                 :data-points test)))
           (attributes-importance* model test-train-data
                                   test-target-data :parallel parallel))))
      cl-ds.alg:array-elementwise
      cl-ds.math:average))


(defun make-confusion-matrix (number-of-classes)
  (make-array (list number-of-classes number-of-classes)
              :element-type 'single-float))


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
          'single-float))


(defun recall (confusion-matrix)
  (let ((folded (fold-general-confusion-matrix confusion-matrix)))
    (coerce (/ (aref folded 1 1)
               (+ (aref folded 0 1)
                  (aref folded 1 1)))
            'single-float)))


(defun specificity (confusion-matrix)
  (let ((folded (fold-general-confusion-matrix confusion-matrix)))
    (coerce (/ (aref folded 0 0)
               (+ (aref folded 0 0)
                  (aref folded 0 1)))
            'single-float)))


(defun precision (confusion-matrix)
  (let ((folded (fold-general-confusion-matrix confusion-matrix)))
    (coerce (/ (aref folded 1 1)
               (+ (aref folded 1 0)
                  (aref folded 1 1)))
            'single-float)))


(defun f1-score (confusion-matrix)
  (let ((precision (precision confusion-matrix))
        (recall (recall confusion-matrix)))
    (coerce (/ (* 2 precision recall)
               (+ precision recall))
            'single-float)))


(defun performance-metric (parameters target predictions &key weights (type :default))
  (performance-metric* parameters type target predictions weights))
