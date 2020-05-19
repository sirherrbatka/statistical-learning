(cl:in-package #:cl-user)

(ql:quickload :cl-grf)
(ql:quickload :vellum)

(defpackage #:mushrooms-example
  (:use #:cl #:cl-grf.aux-package))

(cl:in-package #:mushrooms-example)

(defparameter *data*
  (~> (vellum:copy-from :csv (~>> (asdf:system-source-directory :cl-grf)
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
         (result (cl-grf.data:make-data-matrix (vellum:row-count table)
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
                          (setf (cl-grf.data:mref result index (+ offset encoded)) 1.0d0))
                        (incf index)))
    result))

(defparameter *train-data*
  (encode (vellum:select *data* :columns '(:take-from cap-shape :take-to habitat))))

(defparameter *target-data*
  (lret ((result (cl-grf.data:make-data-matrix (vellum:row-count *data*) 1)))
    (iterate
      (for i from 0 below (vellum:row-count *data*))
      (setf (cl-grf.data:mref result i 0)
            (eswitch ((vellum:at *data* 'class i) :test 'equal)
              ("p" 1.0d0)
              ("e" 0.0d0))))))

(defparameter *training-parameters*
  (make 'cl-grf.algorithms:single-impurity-classification
        :maximal-depth 8
        :minimal-difference 0.0001d0
        :number-of-classes 2
        :minimal-size 10
        :trials-count 80
        :parallel nil))

(defparameter *forest-parameters*
  (make 'cl-grf.forest:random-forest-parameters
        :trees-count 200
        :forest-class 'cl-grf.forest:classification-random-forest
        :parallel t
        :tree-batch-size 10
        :tree-attributes-count 40
        :tree-sample-rate 0.2
        :tree-parameters *training-parameters*))

(defparameter *confusion-matrix*
  (cl-grf.performance:cross-validation *forest-parameters*
                                       4
                                       *train-data*
                                       *target-data*
                                       t))

(print (cl-grf.performance:accuracy *confusion-matrix*)) ; 1.0
