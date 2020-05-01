(cl:in-package #:cl-grf.data)


(declaim (inline attributes-count))
(defun attributes-count (data-matrix)
  (check-type data-matrix data-matrix)
  (array-dimension data-matrix 1))


(declaim (inline data-points-count))
(defun data-points-count (data-matrix)
  (check-type data-matrix data-matrix)
  (array-dimension data-matrix 0))


(defun data-matrix-dimensions (data-matrix)
  (check-type data-matrix data-matrix)
  (array-dimensions data-matrix))


(declaim (inline mref))
(defun mref (data-matrix data-point feature)
  (check-type data-matrix data-matrix)
  (aref data-matrix data-point feature))


(declaim (inline (setf mref)))
(defun (setf mref) (new-value data-matrix data-point feature)
  (check-type data-matrix data-matrix)
  (setf (aref data-matrix data-point feature) new-value))


(defun make-data-matrix (data-points-count attributes-count)
  (check-type data-points-count fixnum)
  (check-type attributes-count fixnum)
  (make-array `(,data-points-count ,attributes-count)
              :initial-element 0.0d0
              :element-type 'double-float))
