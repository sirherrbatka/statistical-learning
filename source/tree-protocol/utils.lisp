(cl:in-package #:statistical-learning.tree-protocol)


(declaim (inline random-uniform))
(defun random-uniform (min max)
  (+ (random (- max min)) min))


(-> data-min/max (statistical-learning.data:data-matrix fixnum)
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


(-> random-test ((simple-array fixnum (*)) statistical-learning.data:data-matrix)
    (values fixnum double-float))
(defun random-test (attributes data)
  "Uses ExtraTree approach."
  (declare (optimize (speed 3) (safety 0)))
  (bind ((attributes-count (length attributes))
         (attribute-index (random attributes-count))
         ((:values min max) (data-min/max data attribute-index))
         (threshold (if (= min max) min (random-uniform min max))))
    (assert (= attributes-count (statistical-learning.data:attributes-count data)))
    (values attribute-index (if (= threshold max) min threshold))))


(-> fill-split-array (statistical-learning.data:data-matrix
                      fixnum double-float
                      sl.opt:split-array)
    (values fixnum fixnum))
(defun fill-split-array (data attribute threshold array)
  (iterate
    (declare (type fixnum right-count left-count i)
             (type boolean rightp)
             (optimize (speed 3) (safety 0)))
    (with right-count = 0)
    (with left-count = 0)
    (for i from 0 below (length array))
    (for rightp = (> (statistical-learning.data:mref data i attribute) threshold))
    (setf (aref array i) rightp)
    (if rightp (incf right-count) (incf left-count))
    (finally (return (values left-count right-count)))))


(-> subsample-array (statistical-learning.data:data-matrix
                     fixnum
                     sl.opt:split-array
                     t
                     (or null fixnum))
    statistical-learning.data:data-matrix)
(defun subsample-array (array length split-array position skipped-column)
  (declare (optimize (speed 3) (safety 0)))
  (cl-ds.utils:cases ((null skipped-column))
    (statistical-learning.data:bind-data-matrix-dimensions
        ((data-points-count attributes-count array))
      (lret ((result (make-array `(,length ,(if (null skipped-column)
                                                attributes-count
                                                (1- attributes-count)))
                                 :element-type 'double-float)))
        (iterate
          (declare (type fixnum j i))
          (with j = 0)
          (for i from 0 below data-points-count)
          (when (eql position (aref split-array i))
            (iterate
              (declare (type fixnum k p))
              (with p = 0)
              (for k from 0 below attributes-count)
              (when (eql skipped-column k)
                (next-iteration))
              (setf (statistical-learning.data:mref result j p)
                    (statistical-learning.data:mref array i k)
                    p (1+ p))
              (finally (assert (= p (array-dimension result 1)))))
            (incf j))
          (finally (assert (= j length))))))))


(defun subsample-vector (vector skipped-position)
  (lret ((result (make-array (1- (length vector))
                             :element-type (array-element-type vector))))
    (iterate
      (for i from 0 below skipped-position)
      (setf (aref result i) (aref vector i)))
    (iterate
      (for i from (1+ skipped-position) below (length vector))
      (setf (aref result (1- i)) (aref vector i)))))
