(cl:in-package #:cl-user)

(ql:quickload '(:vellum :vellum-csv :statistical-learning))

(defpackage #:mushrooms-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:mushrooms-example)

(defvar *data*
  (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                              (merge-pathnames "examples/mushrooms.data"))
                    :includes-header-p nil
                    :columns '((:name class)
                               (:name cap-shape)
                               (:name cap-surface)
                               (:name cap-color)
                               (:name bruises?)
                               (:name odor)
                               (:name gill-attachment)
                               (:name gill-spacing)
                               (:name gill-size)
                               (:name gill-color)
                               (:name stalk-shape)
                               (:name stalk-root)
                               (:name stalk-surface-above-ring)
                               (:name stalk-surface-below-ring)
                               (:name stalk-color-above-ring)
                               (:name stalk-color-below-ring)
                               (:name veil-type)
                               (:name veil-color)
                               (:name ring-number)
                               (:name ring-type)
                               (:name spore-print-color)
                               (:name population)
                               (:name habitat))))

(defparameter *mushroom-types* 2)


(defparameter *aggregation*
  (vellum:aggregate-columns *data*
                            (cl-ds.alg:to-vector)
                            :name 'vector))


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
             (collect (column-encoder-hash-table (vellum:select table :columns `(,i))))))
         (sizes (serapeum:scan #'+ hash-tables :key #'hash-table-count :initial-value 0))
         (total-size (last-elt sizes))
         (result (statistical-learning.data:make-data-matrix (vellum:row-count table)
                                                             total-size))
         (index 0))
    (vellum:transform table
                      (vellum:bind-row ()
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
  (encode (vellum:select *data* :columns (vellum:s (vellum:between :from 'cap-shape)))))

(defparameter *target-data*
  (lret ((result (statistical-learning.data:make-data-matrix (vellum:row-count *data*) 1)))
    (iterate
      (for i from 0 below (vellum:row-count *data*))
      (setf (statistical-learning.data:mref result i 0)
            (eswitch ((vellum:at *data* i 'class) :test 'equal)
              ("p" 1.0d0)
              ("e" 0.0d0))))))

(defparameter *training-parameters*
  (make 'statistical-learning.dt:classification
        :optimized-function (sl.opt:gini-impurity 2)
        :maximal-depth 5
        :minimal-difference 0.0001d0
        :minimal-size 10
        :parallel t
        :splitter (sl.common:lift (make-instance 'sl.tp:random-attribute-splitter)
                                  'sl.tp:random-splitter
                                  :trials-count 80)))

(defparameter *forest-parameters*
  (make 'statistical-learning.ensemble:random-forest
        :trees-count 100
        :parallel t
        :weights-calculator (make-instance 'sl.ensemble:dynamic-weights-calculator)
        :tree-batch-size 5
        :tree-attributes-count 30
        :data-points-sampler (make-instance 'sl.ensemble:weights-based-data-points-sampler
                                            :sampling-rate 0.3)
        :tree-parameters *training-parameters*))

(defparameter *confusion-matrix*
  (statistical-learning.performance:cross-validation *forest-parameters*
                                                     4
                                                     *train-data*
                                                     *target-data*
                                                     :parallel nil))

(print (sl.perf:accuracy *confusion-matrix*)) ; ~0.99

(print (sl.perf:attributes-importance *forest-parameters*
                                      4
                                      *train-data*
                                      *target-data*
                                      :parallel t))
