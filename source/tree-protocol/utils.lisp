(cl:in-package #:statistical-learning.tree-protocol)


(-> wdot (sl.data:double-float-data-matrix
          sl.data:double-float-data-matrix
          fixnum
          fixnum
          (simple-array fixnum (*)))
    double-float)
(defun wdot (first second first-point second-point attributes)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (iterate
    (declare (type fixnum i)
             (type double-float result))
    (with result = 0.0d0)
    (for i from 0 below (length attributes))
    (for attribute = (aref attributes i))
    (incf result
          (* (sl.data:mref first first-point attribute)
             (sl.data:mref second second-point i)))
    (finally (return result))))
