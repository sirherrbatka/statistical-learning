(cl:in-package #:cl-user)

(ql:quickload '(:vellum :vellum-csv :statistical-learning))

(defpackage #:airfoil-noise-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:airfoil-noise-example)

(defvar *data*
  (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                              (merge-pathnames "examples/airfoil_self_noise.dat"))
                    :separator #\tab
                    :includes-header-p nil
                    :columns '((:name frequency :type float)
                               (:name angle :type float)
                               (:name chord-length :type float)
                               (:name velocity :type float)
                               (:name displacement :type float)
                               (:name sound :type float))))

(defvar *train-data*
  (vellum:to-matrix (vellum:select *data* :columns (vellum:s (vellum:between :to 'sound)))
                    :element-type 'double-float))

(defvar *target-data*
  (vellum:to-matrix (vellum:select *data* :columns '(sound))
                    :element-type 'double-float))

(defparameter *training-parameters*
  (make 'statistical-learning.dt:regression
        :optimized-function (sl.opt:squared-error)
        :maximal-depth 8
        :minimal-difference 0.0001d0
        :minimal-size 5
        :splitter (sl.common:lift (make-instance 'sl.tp:random-attribute-splitter)
                                  'sl.tp:random-splitter
                                  :trials-count 80)
        :parallel t))

(defparameter *forest-parameters*
  (make 'statistical-learning.ensemble:random-forest
        :trees-count 250
        :parallel t
        :tree-batch-size 25
        :pruning (make-instance 'sl.omp:orthogonal-matching-pursuit :number-of-trees-selected 50)
        :tree-attributes-count 5
        :data-points-sampler (make-instance 'sl.ensemble:weights-based-data-points-sampler
                                            :sampling-rate 0.2)
        :tree-parameters *training-parameters*))

(defparameter *mean-error*
  (statistical-learning.performance:cross-validation *forest-parameters*
                                                     4
                                                     *train-data*
                                                     *target-data*
                                                     :parallel nil))


(format t "~3$~%" *mean-error*) ; ~19.00 (squared error)


(defparameter *forest*
  (sl.mp:make-supervised-model *forest-parameters*
                               *train-data*
                               *target-data*
                               :parallel nil))


(defparameter *som-model*
  (sl.mp:make-unsupervised-model
   (make-instance 'sl.som:random-forest-self-organizing-map
                  :parallel nil
                  :forest *forest*
                  :random-ranges (vellum:pipeline (*data*)
                                   (cl-ds.alg:on-each
                                    (vellum:bind-row ()
                                      (~> (vellum:between :to 'sound)
                                          vellum:vs
                                          cl-ds.alg:to-vector)))
                                   cl-ds.alg:array-elementwise
                                   (cl-ds.alg:extrema #'<)
                                   (cl-ds.alg:to-list :key (lambda (x) (list (car x) (cdr x)))))
                  :grid-dimensions '(10 10)
                  :number-of-iterations 1000
                  :initial-alpha 1.0d0
                  :decay sl.som:<hill-decay>)
   *train-data*))

(defparameter *positions* (sl.mp:predict *som-model* *train-data*))

(format t "~3$~%"
        (statistical-learning.performance:cross-validation
         (make 'statistical-learning.ensemble:gradient-boost-ensemble
               :trees-count 500
               :parallel t
               :tree-batch-size 10
               :shrinkage 0.1d0
               :tree-attributes-count 5
               :data-points-sampler (make 'sl.ensemble:weights-based-data-points-sampler
                                          :sampling-rate 0.5)
               :tree-parameters (make 'sl.gbt:regression
                                      :optimized-function (sl.opt:squared-error)
                                      :maximal-depth 5
                                      :minimal-size 5
                                      :minimal-difference 0.00001d0
                                      :splitter (sl.common:lift (make-instance 'sl.tp:random-attribute-splitter)
                                                                'sl.tp:random-splitter
                                                                :trials-count 15)
                                      :parallel t))
         4
         *train-data*
         *target-data*
         :parallel nil)) ; ~3.5 (also squared error, obviously a lot better)
