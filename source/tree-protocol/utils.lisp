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
             (type boolean right-p)
             (optimize (speed 3) (safety 0)))
    (with right-count = 0)
    (with left-count = 0)
    (for i from 0 below (length array))
    (for right-p = (> (statistical-learning.data:mref data i attribute) threshold))
    (setf (aref array i) right-p)
    (if right-p (incf right-count) (incf left-count))
    (finally (return (values left-count right-count)))))


(-> subsample-array (statistical-learning.data:data-matrix
                     fixnum
                     sl.opt:split-array
                     boolean
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
          (when (eq position (aref split-array i))
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


(defmethod make-simple-node
    (parameters
     training-parameters
     split-array
     score
     length
     position
     parallel
     training-state
     new-attributes
     attribute-index)
  (if (not parallel)
      #1=(let* ((old-target-data (target-data training-state))
                (old-training-data (training-data training-state))
                (weights (sl.tp:weights training-state))
                (new-state
                  (sl.tp:training-state-clone
                   training-state
                   :depth (~> training-parameters depth 1+)
                   :loss score
                   :weights (if (null weights)
                                nil
                                (subsample-array weights length
                                                 split-array position
                                                 nil))
                   :training-data (subsample-array old-training-data
                                                   length split-array
                                                   position attribute-index)
                   :target-data (subsample-array old-target-data
                                                 length split-array
                                                 position nil)
                   :attribute-indexes new-attributes))
                (new-leaf (make-leaf new-state)))
           (setf training-state nil
                 split-array nil) ; so it can be gced
           (split new-state new-leaf))
      (lparallel:future #1#)))
