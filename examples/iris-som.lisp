(ql:quickload :statistical-learning)
(ql:quickload :vellum)

(cl:in-package #:cl-user)

(defpackage #:iris-som-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:iris-som-example)

;; loading data set
(defvar *data*
  (~> (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                                  (merge-pathnames "examples/iris.data"))
                        :header nil)
      (vellum:to-table :columns '((:alias sepal-length :type float)
                                  (:alias sepal-width :type float)
                                  (:alias petal-length :type float)
                                  (:alias petal-width :type float)
                                  (:alias class :type string)))
      (vellum:select :columns '(:take-from sepal-length :take-to petal-width))))

;; scale data to 0.0 : 1.0 range
(defvar *extrema*
  (vellum:aggregate-rows *data*
                         'sepal-length ((cl-ds.alg:extrema #'<) :skip-nulls t)
                         'sepal-width ((cl-ds.alg:extrema #'<) :skip-nulls t)
                         'petal-length ((cl-ds.alg:extrema #'<) :skip-nulls t)
                         'petal-width ((cl-ds.alg:extrema #'<) :skip-nulls t)))

(defun scale (value extrema)
  (/ (- value (car extrema))
     (cdr extrema)))

(defvar *training-data*
  (~> *data*
      (vellum:transform
       (vellum:body (sepal-length sepal-width petal-length petal-width)
         (setf sepal-length (scale sepal-length (vellum:at *extrema* 0 'sepal-length))
               sepal-width (scale sepal-width (vellum:at *extrema* 0 'sepal-width))
               petal-length (scale petal-length (vellum:at *extrema* 0 'petal-length))
               petal-width (scale petal-width (vellum:at *extrema* 0 'petal-width)))))
      (vellum:to-matrix :element-type 'double-float)))

(defparameter *training-parameters*
  (make 'sl.som:self-organizing-map
        :grid-dimensions '(20 20)
        :number-of-iterations 10000
        :initial-alpha 0.6d0
        :initial-sigma 10.0d0
        :decay sl.som:<linear-decay>
        :parallel nil))

(defparameter *model*
  (sl.mp:make-unsupervised-model *training-parameters* *training-data*))
