(ql:quickload '(:vellum :vellum-csv :statistical-learning))

(cl:in-package #:cl-user)

(defpackage #:iris-som-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:iris-som-example)

;; loading data set
(defvar *data*
  (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                              (merge-pathnames "examples/iris.data"))
                    :includes-header-p nil
                    :columns '((:name sepal-length :type float)
                               (:name sepal-width :type float)
                               (:name petal-length :type float)
                               (:name petal-width :type float)
                               (:name class :type string))))

(defvar *ranges*
  (vellum:pipeline (*data*)
    (cl-ds.alg:on-each
     (vellum:bind-row ()
       (~> (vellum:between :to 'class)
           vellum:vs
           cl-ds.alg:to-vector)))
    cl-ds.alg:array-elementwise
    (cl-ds.alg:extrema #'<)
    (cl-ds.alg:to-list :key (lambda (x) (list (car x) (cdr x))))))

(defvar *training-data*
  (~> *data*
      (vellum:select :columns (vellum:s (vellum:between :to 'class)))
      (vellum:to-matrix :element-type 'double-float)))

(defparameter *training-parameters*
  (make 'sl.som:self-organizing-map
        :grid-dimensions '(4 4)
        :random-ranges *ranges*
        :number-of-iterations 10000
        :initial-alpha 0.3d0
        :decay sl.som:<linear-decay>
        :parallel nil))

(defparameter *model*
  (sl.mp:make-unsupervised-model *training-parameters* *training-data*))

(defparameter *positions* (sl.mp:predict *model* *training-data*))

(defparameter *with-positions*
  (~> *positions* cl-ds.utils:unfold-table (batches 2)
      (cl-ds.alg:on-each #'vector :key (curry #'map 'list #'truncate))
      (vellum:to-table :columns '((:name position)))
      (list *data* _)
      vellum:hstack))

(defparameter *classes* (vellum:pipeline (*with-positions*)
                          (cl-ds.alg:group-by :test 'equal
                                              :key (vellum:brr position))
                          (cl-ds.alg:group-by :test 'equal
                                              :key (vellum:brr class))
                          cl-ds.alg:count-elements
                          (vellum:to-table :columns '((:name position)
                                                      (:alas class)
                                                      (:name count)))
                          (vellum:order-by 'position
                                           (curry #'cl-ds.utils:lexicographic-compare
                                                  #'< #'=))))
