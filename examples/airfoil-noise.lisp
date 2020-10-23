(cl:in-package #:cl-user)

(ql:quickload :statistical-learning)
(ql:quickload :vellum)

(defpackage #:airfoil-noise-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:airfoil-noise-example)

(defvar *data*
  (~> (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                                  (merge-pathnames "examples/airfoil_self_noise.dat"))
                        :header nil
                        :separator #\tab)
      (vellum:to-table :columns '((:alias frequency :type float)
                                  (:alias angle :type float)
                                  (:alias chord-length :type float)
                                  (:alias velocity :type float)
                                  (:alias displacement :type float)
                                  (:alias sound :type float)))))

(defvar *train-data*
  (vellum:to-matrix (vellum:select *data* :columns '(:take-to displacement))
                    :element-type 'double-float))

(defvar *target-data*
  (vellum:to-matrix (vellum:select *data* :columns '(:v sound))
                    :element-type 'double-float))

(defparameter *training-parameters*
  (make 'statistical-learning.dt:regression
        :optimized-function (sl.opt:squared-error)
        :maximal-depth 4
        :minimal-difference 0.0001d0
        :minimal-size 5
        :trials-count 50
        :parallel nil))

(defparameter *forest-parameters*
  (make 'statistical-learning.ensemble:random-forest
        :trees-count 250
        :parallel nil
        :tree-batch-size 250
        :tree-attributes-count 5
        :tree-sample-rate 0.2
        :tree-parameters *training-parameters*))

(defparameter *mean-error*
  (statistical-learning.performance:cross-validation *forest-parameters*
                                                     4
                                                     *train-data*
                                                     *target-data*
                                                     :parallel nil))

(print *mean-error*) ; 19.14110529750509d0 (squared error)

(print (statistical-learning.performance:cross-validation
        (make 'statistical-learning.ensemble:gradient-boost-ensemble
              :trees-count 500
              :parallel nil
              :tree-batch-size 10
              :shrinkage 0.1d0
              :tree-attributes-count 5
              :tree-sample-rate 0.5
              :tree-parameters (make 'sl.gbt:regression
                                     :optimized-function (sl.opt:squared-error)
                                     :maximal-depth 5
                                     :minimal-size 5
                                     :minimal-difference 0.00001d0
                                     :trials-count 15
                                     :parallel t))
        4
        *train-data*
        *target-data*
        :parallel nil)) ; 3.97 (also squared error, obviously a lot better)
