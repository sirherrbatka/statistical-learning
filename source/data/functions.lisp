(cl:in-package #:cl-grf.data)


(declaim (inline attributes-count))
(defun attributes-count (data-matrix)
  (check-data-points data-matrix)
  (array-dimension data-matrix 1))


(declaim (inline data-points-count))
(defun data-points-count (data-matrix)
  (check-data-points data-matrix)
  (array-dimension data-matrix 0))


(defun data-matrix-dimensions (data-matrix)
  (check-data-points data-matrix)
  (array-dimensions data-matrix))


(declaim (inline mref))
(defun mref (data-matrix data-point attribute)
  (check-data-points data-matrix)
  (aref data-matrix data-point attribute))


(declaim (inline (setf mref)))
(defun (setf mref) (new-value data-matrix data-point attribute)
  (check-data-points data-matrix)
  (setf (aref data-matrix data-point attribute) new-value))


(-> make-data-matrix (fixnum fixnum &optional double-float)
    data-matrix)
(defun make-data-matrix (data-points-count attributes-count &optional (initial-element 0.0d0))
  (check-type data-points-count fixnum)
  (check-type attributes-count fixnum)
  (assert (> attributes-count 0))
  (assert (> data-points-count 0))
  (make-array `(,data-points-count ,attributes-count)
              :initial-element initial-element
              :element-type 'double-float))


(-> sample (data-matrix &key
                        (:data-points (or null (simple-array fixnum (*))))
                        (:attributes (or null (simple-array fixnum (*)))))
    data-matrix)
(defun sample (data-matrix &key data-points attributes)
  (declare (optimize (speed 3) (safety 0)))
  (check-data-points data-matrix)
  (assert (or data-points attributes))
  (cl-ds.utils:cases ((null attributes)
                      (null data-points))
    (iterate
      (declare (type fixnum i attributes-count data-points-count))
      (with attributes-count = (if (null attributes)
                                   (attributes-count data-matrix)
                                   (length attributes)))
      (with data-points-count = (if (null data-points)
                                    (data-points-count data-matrix)
                                    (length data-points)))
      (with result = (make-data-matrix data-points-count
                                       attributes-count))
      (for i from 0 below data-points-count)
      (iterate
        (declare (type fixnum j))
        (for j from 0 below attributes-count)
        (setf (mref result i j)
              (mref data-matrix
                    (if (null data-points) i (aref data-points i))
                    (if (null attributes) j (aref attributes j)))))
      (finally (return result)))))


(declaim (inline map-data-matrix))
(defun map-data-matrix (function data-matrix &optional in-place)
  (declare (optimize (speed 3) (safety 0)))
  (check-type data-matrix cl-grf.data:data-matrix)
  (lret ((result (if in-place
                     data-matrix
                     (make-data-matrix
                      (data-points-count data-matrix)
                      (attributes-count data-matrix)))))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below (array-total-size data-matrix))
      (setf (row-major-aref result i)
            (funcall function (row-major-aref data-matrix i))))))


(defun make-data-matrix-like (data-matrix &optional (initial-element 0.0d0))
  (check-type data-matrix data-matrix)
  (make-array (array-dimensions data-matrix)
              :element-type 'double-float
              :initial-element initial-element))


(declaim (inline reduce-data-points))
(defun reduce-data-points (function data-matrix)
  (declare (optimize (speed 3) (safety 0)))
  (check-type data-matrix data-matrix)
  (iterate
    (declare (type fixnum i attributes-count data-points-count))
    (with function = (ensure-function function))
    (with data-points-count = (data-points-count data-matrix))
    (with attributes-count = (attributes-count data-matrix))
    (with result = (make-data-matrix 1 attributes-count))
    (for i from 0 below data-points-count)
    (iterate
      (declare (type fixnum j))
      (for j from 0 below attributes-count)
      (setf (mref result 0 j)
            (the double-float (funcall function
                                       (mref result 0 j)
                                       (mref data-matrix i j)))))
    (finally (return result))))
