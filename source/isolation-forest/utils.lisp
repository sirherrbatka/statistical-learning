(cl:in-package #:statistical-learning.isolation-forest)


(-> scale-double-float
    (double-float double-float double-float double-float double-float)
    double-float)
(defun scale-double-float (value range scale-min min max)
  (if (= max min)
      value
      (+ (* range (/ (- value min) (- max min)))
         scale-min)))


(-> wdot (sl.data:double-float-data-matrix
          sl.data:double-float-data-matrix
          fixnum
          fixnum
          (simple-array fixnum (*)))
    double-float)
(defun wdot (first second first-point second-point attributes)
  (declare (optimize (speed 0) (safety 3) (debug 3)))
  (iterate
    (for i from 0 below (length attributes))
    (for attribute = (aref attributes i))
    (sum (* (sl.data:mref first first-point attribute)
            (sl.data:mref second second-point i)))))


(defun rightp (split-point attributes data-point data)
  (declare (type sl.data:double-float-data-matrix data)
           (type (simple-array fixnum (*)) attributes)
           (type fixnum data-point))
  (let ((dot-product (split-point-dot-product split-point))
        (normals (split-point-normals split-point)))
    (< (wdot data normals data-point 0 attributes)
       dot-product)))


(defun reduce-data-points (data samples attributes function)
  (iterate
    (with function = (ensure-function function))
    (with data-points-count = (length samples))
    (with attributes-count = (length attributes))
    (with result = (sl.data:make-data-matrix 1 attributes-count))
    (for i from 0 below attributes-count)
    (for attribute = (aref attributes i))
    (setf (sl.data:mref result 0 i)
          (sl.data:mref data (aref samples 0) attribute))
    (iterate
      (for j from 1 below data-points-count)
      (for k = (aref samples j))
      (setf (sl.data:mref result k i)
            (funcall function
                     (sl.data:mref result k i)
                     (sl.data:mref data k attribute))))
    (finally (return result))))


(defun calculate-mins (data-matrix samples attributes)
  (reduce-data-points data-matrix
                      samples
                      attributes
                      #'min))


(defun calculate-maxs (data-matrix samples attributes)
  (reduce-data-points data-matrix
                      samples
                      attributes
                      #'max))


(defun make-normals (count)
  (iterate
    (with result = (sl.data:make-data-matrix 1 count))
    (for i from 0 below count)
    (setf (sl.data:mref result 0 i) (sl.common:gauss-random))
    (finally (return result))))
