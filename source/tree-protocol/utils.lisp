(cl:in-package #:statistical-learning.tree-protocol)


(declaim (inline random-uniform))
(defun random-uniform (min max)
  (+ (random (- max min)) min))


(-> data-min/max (sl.data:double-float-data-matrix fixnum)
    (values double-float double-float))
(defun data-min/max (data attribute)
  (declare (type statistical-learning.data:data-matrix data)
           (type fixnum attribute)
           (optimize (speed 3) (safety 0)))
  (iterate
    (declare (type double-float min max element)
             (type fixnum i))
    (with min = (statistical-learning.data:mref data 0 attribute))
    (with max = min)
    (with length = (statistical-learning.data:data-points-count data))
    (for i from 1 below length)
    (for element = (statistical-learning.data:mref data i attribute))
    (cond ((< max element) (setf max element))
          ((> min element) (setf min element)))
    (finally (return (values min max)))))


(-> random-test (fundamental-tree-training-parameters
                 (simple-array fixnum (*))
                 sl.data:double-float-data-matrix)
    (values fixnum double-float))
(defun random-test (parameters attributes data)
  "Uses ExtraTree approach."
  (declare (optimize (speed 3) (safety 0))
           (ignore parameters))
  (bind ((attributes-count (length attributes))
         (attribute-index (random attributes-count))
         ((:values min max) (data-min/max data attribute-index))
         (threshold (if (= min max) min (random-uniform min max))))
    (assert (= attributes-count (statistical-learning.data:attributes-count data)))
    (values attribute-index (if (= threshold max) min threshold))))


(-> fill-split-array (fundamental-tree-training-parameters
                      statistical-learning.data:data-matrix
                      fixnum double-float
                      sl.data:split-vector)
    (values fixnum fixnum))
(defun fill-split-array (parameters
                         data attribute threshold array)
  (declare (ignore parameters)
           (optimize (speed 3) (safety 0)))
  (iterate
    (declare (type fixnum right-count left-count i)
             (type boolean rightp))
    (with right-count = 0)
    (with left-count = 0)
    (for i from 0 below (length array))
    (for rightp = (> (statistical-learning.data:mref data i attribute) threshold))
    (setf (aref array i) rightp)
    (if rightp (incf right-count) (incf left-count))
    (finally (return (values left-count right-count)))))


(defun subsample-vector (vector skipped-position)
  (lret ((result (make-array (1- (length vector))
                             :element-type (array-element-type vector))))
    (iterate
      (for i from 0 below skipped-position)
      (setf (aref result i) (aref vector i)))
    (iterate
      (for i from (1+ skipped-position) below (length vector))
      (setf (aref result (1- i)) (aref vector i)))))
