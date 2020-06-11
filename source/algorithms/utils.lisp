(cl:in-package #:cl-grf.algorithms)


(declaim (inline random-uniform))
(defun random-uniform (min max)
  (+ (random (- max min)) min))


(-> data-min/max (cl-grf.data:data-matrix fixnum)
    (values double-float double-float))
(defun data-min/max (data attribute)
  (declare (type cl-grf.data:data-matrix data)
           (type fixnum attribute)
           (optimize (speed 3) (safety 0)))
  (iterate
    (declare (type double-float min max element)
             (type fixnum i))
    (with min = (cl-grf.data:mref data 0 attribute))
    (with max = min)
    (with length = (cl-grf.data:data-points-count data))
    (for i from 1 below length)
    (for element = (cl-grf.data:mref data i attribute))
    (cond ((< max element) (setf max element))
          ((> min element) (setf min element)))
    (finally (return (values min max)))))


(-> random-test ((simple-array fixnum (*)) cl-grf.data:data-matrix)
    (values fixnum double-float))
(defun random-test (attributes data)
  "Uses ExtraTree approach."
  (declare (optimize (speed 3) (safety 0)))
  (bind ((attributes-count (length attributes))
         (attribute-index (random attributes-count))
         ((:values min max) (data-min/max data attribute-index))
         (threshold (if (= min max) min (random-uniform min max))))
    (assert (= attributes-count (cl-grf.data:attributes-count data)))
    (values attribute-index (if (= threshold max) min threshold))))


(-> fill-split-array (cl-grf.data:data-matrix
                      fixnum double-float
                      (simple-array boolean (*)))
    (values fixnum fixnum))
(defun fill-split-array (data attribute threshold array)
  (iterate
    (declare (type fixnum right-count left-count i)
             (type boolean right-p)
             (optimize (speed 3) (safety 0)))
    (with right-count = 0)
    (with left-count = 0)
    (for i from 0 below (length array))
    (for right-p = (> (cl-grf.data:mref data i attribute) threshold))
    (setf (aref array i) right-p)
    (if right-p (incf right-count) (incf left-count))
    (finally (return (values left-count right-count)))))


(-> shannon-entropy (double-float) double-float)
(defun shannon-entropy (probability)
  (declare (optimize (speed 3) (safety 0)))
  (if (zerop probability)
      0.0d0
      (- (* probability (the double-float (log probability))))))


(-> vector-impurity ((simple-array double-float *)) double-float)
(defun vector-impurity (sums)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((length (array-total-size sums))
         (grand-total (iterate
                        (declare (type fixnum i)
                                 (type double-float result))
                        (with result = 0.0d0)
                        (for i from 0 below length)
                        (incf result (row-major-aref sums i))
                        (finally (return result)))))
    (declare (type double-float grand-total))
    (if (zerop grand-total)
        0.0d0
        (iterate
          (declare (type fixnum length i)
                   (type double-float impurity p))
          (with impurity = 0.0d0)
          (for i from 0 below length)
          (for sum = (row-major-aref sums i))
          (when (zerop sum)
            (next-iteration))
          (for p = (/ sum grand-total))
          (incf impurity (* p p))
          (finally (return (- 1.0d0 impurity)))))))


(-> split-impurity (single-impurity-classification
                   (simple-array boolean (*))
                   cl-grf.data:data-matrix)
    (values double-float double-float))
(defun split-impurity (parameters split-array target-data)
  (iterate
    (declare (type fixnum length i)
             (type (simple-array double-float (*))
                   left-sums right-sums)
             (optimize (speed 3) (safety 1) (debug 1)))
    (with number-of-classes = (number-of-classes parameters))
    (with left-sums = (make-array number-of-classes
                                  :initial-element 0.0d0
                                  :element-type 'double-float))
    (with right-sums = (make-array number-of-classes
                                   :initial-element 0.0d0
                                   :element-type 'double-float))
    (with length = (cl-grf.data:data-points-count target-data))
    (for i from 0 below length)
    (for right-p = (aref split-array i))
    (for target = (truncate (cl-grf.data:mref target-data i 0)))
    (if right-p
        (incf (aref right-sums target) 1.0d0)
        (incf (aref left-sums target) 1.0d0))
    (finally
     (return (values (vector-impurity left-sums)
                     (vector-impurity right-sums))))))


(defun total-impurity (parameters target-data)
  (let* ((length (cl-grf.data:data-points-count target-data))
         (split-array (make-array length :element-type 'boolean
                                         :initial-element nil)))
    (nth-value 0 (split-impurity parameters split-array target-data))))


(-> subsample-array (cl-grf.data:data-matrix
                     fixnum (simple-array boolean (*))
                     boolean
                     (or null fixnum))
    cl-grf.data:data-matrix)
(defun subsample-array (array length split-array position skipped-column)
  (declare (optimize (speed 3) (safety 0)))
  (cl-ds.utils:cases ((null skipped-column))
    (cl-grf.data:bind-data-matrix-dimensions
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
              (setf (cl-grf.data:mref result j p)
                    (cl-grf.data:mref array i k)
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


(defgeneric make-simple-node* (parameters split-array shannon-entropy
                               length position parallel training-state
                               new-attributes attribute-index))


(defun make-simple-node (split-array shannon-entropy length
                         position parallel training-state
                         new-attributes attribute-index)
  (make-simple-node* (cl-grf.tp:training-parameters training-state)
                     split-array
                     shannon-entropy
                     length
                     position
                     parallel
                     training-state
                     new-attributes
                     attribute-index))


(defmethod make-simple-node*
    ((parameters scored-training)
     split-array
     score
     length
     position
     parallel
     training-state
     new-attributes
     attribute-index)
  (if (not parallel)
      #1=(let* ((old-target-data (cl-grf.tp:target-data training-state))
                (old-training-data (cl-grf.tp:training-data training-state))
                (new-state
                  (cl-grf.tp:training-state-clone
                   training-state
                   :training-data (subsample-array old-training-data
                                                   length split-array
                                                   position attribute-index)
                   :target-data (subsample-array old-target-data
                                                 length split-array
                                                 position nil)
                   :attribute-indexes new-attributes))
                (new-leaf (cl-grf.tp:make-leaf new-state)))
           (setf training-state nil
                 split-array nil) ; so it can be gced
           (incf (cl-grf.tp:depth new-state))
           (let ((split-candidate (cl-grf.tp:split new-state new-leaf)))
             (if (null split-candidate)
                 new-leaf
                 split-candidate)))
      (lparallel:future #1#)))


(defun sum-matrices (matrix result)
  (iterate
    (for i from 0 below (array-total-size matrix))
    (incf (row-major-aref result i)
          (row-major-aref matrix i)))
  result)


(declaim (inline square))
(defun square (x)
  (* x x))
