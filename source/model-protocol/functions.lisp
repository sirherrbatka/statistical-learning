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
