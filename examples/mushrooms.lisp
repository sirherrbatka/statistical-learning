(cl:in-package #:cl-user)

(ql:quickload :statistical-learning)
(ql:quickload :vellum)

(defpackage #:mushrooms-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:mushrooms-example)

(defvar *data*
  (~> (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                                  (merge-pathnames "examples/mushrooms.data"))
                        :header nil)
      (vellum:to-table :columns '((:alias class)
                                  (:alias cap-shape)
                                  (:alias cap-surface)
                                  (:alias cap-color)
                                  (:alias bruises?)
                                  (:alias odor)
                                  (:alias gill-attachment)
                                  (:alias gill-spacing)
                                  (:alias gill-size)
                                  (:alias gill-color)
                                  (:alias stalk-shape)
                                  (:alias stalk-root)
                                  (:alias stalk-surface-above-ring)
                                  (:alias stalk-surface-below-ring)
                                  (:alias stalk-color-above-ring)
                                  (:alias stalk-color-below-ring)
                                  (:alias veil-type)
                                  (:alias veil-color)
                                  (:alias ring-number)
                                  (:alias ring-type)
                                  (:alias spore-print-color)
                                  (:alias population)
                                  (:alias habitat)))))

(defparameter *mushroom-types* 2)

(defun column-encoder-hash-table (input)
  (vellum:with-table (input)
    (~> (cl-ds.alg:on-each input
                           (lambda (_)
                             (declare (ignore _))
                             (vellum:rr 0)))
        (cl-ds.alg:distinct :test #'equal)
        (cl-ds.alg:enumerate :test 'equal))))


(defun encode (table)
  (let* ((column-count (vellum:column-count table))
         (hash-tables
           (iterate
             (for i from 0 below column-count)
             (collect (column-encoder-hash-table (vellum:select table :columns `(:v ,i))))))
         (sizes (serapeum:scan #'+ hash-tables :key #'hash-table-count :initial-value 0))
         (total-size (last-elt sizes))
         (result (statistical-learning.data:make-data-matrix (vellum:row-count table)
                                               total-size))
         (index 0))
    (vellum:transform table
                      (vellum:body ()
                        (iterate
                          (for i from 0 below column-count)
                          (for offset in sizes)
                          (for hash-table in hash-tables)
                          (for v = (vellum:rr i))
                          (for encoded = (gethash v hash-table))
                          (setf (statistical-learning.data:mref result index (+ offset encoded)) 1.0d0))
                        (incf index)))
    result))

(defparameter *train-data*
  (encode (vellum:select *data* :columns '(:take-from cap-shape :take-to habitat))))

(defparameter *target-data*
  (lret ((result (statistical-learning.data:make-data-matrix (vellum:row-count *data*) 1)))
    (iterate
      (for i from 0 below (vellum:row-count *data*))
      (setf (statistical-learning.data:mref result i 0)
            (eswitch ((vellum:at *data* 'class i) :test 'equal)
              ("p" 1.0d0)
              ("e" 0.0d0))))))

(defparameter *training-parameters*
  (make 'statistical-learning.dt:classification
        :optimized-function (sl.opt:gini-impurity 2)
        :maximal-depth 8
        :minimal-difference 0.0001d0
        :minimal-size 10
        :trials-count 80
        :parallel nil))

(defparameter *forest-parameters*
  (make 'statistical-learning.ensemble:random-forest
        :trees-count 200
        :parallel t
        :tree-batch-size 5
        :tree-attributes-count 35
        :tree-sample-rate 0.2
        :tree-parameters *training-parameters*))


(defparameter *confusion-matrix*
  (statistical-learning.performance:cross-validation *forest-parameters*
                                                     4
                                                     *train-data*
                                                     *target-data*
                                                     t))

(print (statistical-learning.performance:accuracy *confusion-matrix*)) ; 1.0

(print (statistical-learning.performance:attributes-importance *forest-parameters*
                                                               4
                                                               *train-data*
                                                               *target-data*
                                                               t))
