(cl:in-package #:statistical-learning.tree-protocol)


(declaim (inline random-uniform))
(defun random-uniform (min max)
  (+ (random (- max min)) min))


(-> data-min/max (sl.data:double-float-data-matrix
                  fixnum
                  (simple-array fixnum (*)))
    (values double-float double-float))
(defun data-min/max (data attribute data-points)
  (declare (type statistical-learning.data:data-matrix data)
           (type fixnum attribute)
           (optimize (speed 3) (safety 0)))
  (iterate
    (declare (type double-float min max element)
             (type fixnum i))
    (with min = (sl.data:mref data (aref data-points 0) attribute))
    (with max = min)
    (with length = (length data-points))
    (for i from 1 below length)
    (for element = (sl.data:mref data (aref data-points i) attribute))
    (cond ((< max element) (setf max element))
          ((> min element) (setf min element)))
    (finally (return (values min max)))))
