(cl:in-package #:cl-user)

(ql:quickload :cl-grf)
(ql:quickload :vellum)

(defpackage #:airfoil-noise-example
  (:use #:cl #:cl-grf.aux-package))

(cl:in-package #:airfoil-noise-example)

(defparameter *data*
  (~> (vellum:copy-from :csv (~>> (asdf:system-source-directory :cl-grf)
                                  (merge-pathnames "examples/airfoil_self_noise.dat"))
                        :header nil
                        :separator #\tab)
      (vellum:to-table :columns '((:alias frequency :type float)
                                  (:alias angle :type float)
                                  (:alias chord-length :type float)
                                  (:alias velocity :type float)
                                  (:alias displacement :type float)
                                  (:alias sound :type float)))))

(defparameter *train-data*
  (vellum:to-matrix (vellum:select *data* :columns '(:take-to displacement))
                    :element-type 'double-float))

(defparameter *target-data*
  (vellum:to-matrix (vellum:select *data* :columns '(:v sound))
                    :element-type 'double-float))

(defparameter *training-parameters*
  (make 'cl-grf.algorithms:regression
        :maximal-depth 5
        :minimal-difference 0.0001d0
        :minimal-size 10
        :trials-count 80
        :parallel nil))

(defparameter *forest-parameters*
  (make 'cl-grf.forest:regression-random-forest-parameters
        :trees-count 200
        :parallel nil
        :tree-batch-size 10
        :tree-attributes-count 40
        :tree-sample-rate 0.2
        :tree-parameters *training-parameters*))

(defparameter *mean-error*
  (cl-grf.performance:cross-validation *forest-parameters*
                                       4
                                       *train-data*
                                       *target-data*
                                       nil))

(print *mean-error*) ; 1.0

(print (cl-grf.performance:attributes-importance *forest-parameters*
                                                 4
                                                 *train-data*
                                                 *target-data*
                                                 t))
