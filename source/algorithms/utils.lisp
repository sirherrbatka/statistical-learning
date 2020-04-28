(cl:in-package #:cl-grf.algorithms)


(declaim (inline random-uniform))
(defun random-uniform (min max)
  (+ (random (- max min)) min))


(-> data-min/max ((simple-array double-float (* *)) fixnum)
    (values double-float double-float))
(defun data-min/max (data attribute)
  (declare (type (simple-array double-float (* *)) data)
           (type fixnum attribute)
           (optimize (speed 3) (safety 0)))
  (iterate
    (declare (type double-float min max element)
             (type fixnum i))
    (with min = (aref data 0 attribute))
    (with max = min)
    (with length = (array-dimension data 0))
    (for i from 1 below length)
    (for element = (aref data i attribute))
    (cond ((< max element) (setf max element))
          ((> min element) (setf min element)))
    (finally (return (values min max)))))


(-> random-test ((simple-array double-float (* *)))
    (values fixnum double-float))
(defun random-test (data)
  (declare (optimize (speed 3) (safety 0)))
  (bind ((attribute (random (array-dimension data 1)))
         ((:values min max) (data-min/max data attribute))
         (threshold (if (= (the double-float min) (the double-float max))
                        min
                        (random-uniform min max))))
    (values attribute threshold)))


(-> fill-split-array ((simple-array double-float (* *))
                      fixnum double-float
                      (simple-array boolean (*)))
    (values fixnum fixnum))
(defun fill-split-array (data attribute threshold array)
  (iterate
    (declare (type fixnum true-count false-count i)
             (type boolean check)
             (optimize (speed 3) (safety 0)))
    (with true-count = 0)
    (with false-count = 0)
    (for i from 0 below (array-dimension array 0))
    (for check = (> (aref data i attribute) threshold))
    (setf (aref array i) check)
    (if check (incf true-count) (incf false-count))
    (finally (return (values false-count true-count)))))


(-> shannon-entropy (double-float) double-float)
(defun shannon-entropy (probability)
  (declare (optimize (speed 3) (safety 0)))
  (if (zerop probability)
      0.0d0
      (- (* probability (the double-float (log probability))))))


(-> split-entropy ((simple-array boolean (*))
                   (simple-array double-float (* *)))
    (values double-float double-float))
(defun split-entropy (split-array target-data)
  (iterate
    (declare (type fixnum length j)
             (type double-float
                   left-entropy right-entropy)
             (optimize (speed 0) (safety 0)
                       (debug 3)))
    (with left-entropy = 0.0d0)
    (with right-entropy = 0.0d0)
    (with length = (array-dimension target-data 0))
    (with number-of-classes = (array-dimension target-data 1))
    (for j from 0 below number-of-classes)
    (iterate
      (declare (type fixnum i left-count right-count)
               (type double-float left-sum right-sum))
      (with left-sum = 0.0d0)
      (with right-sum = 0.0d0)
      (with left-count = 0)
      (with right-count = 0)
      (for i from 0 below length)
      (for right-p = (aref split-array i))
      (if right-p
          (progn
            (incf right-sum (aref target-data i j))
            (incf right-count))
          (progn
            (incf left-sum (aref target-data i j))
            (incf left-count)))
      (finally
       (unless (zerop left-count)
         (let ((left-probability (/ left-sum left-count)))
           (incf left-entropy (shannon-entropy left-probability))
           (incf left-entropy (shannon-entropy (- 1.0d0 left-probability)))))
       (unless (zerop right-count)
         (let ((right-probability (/ right-sum right-count)))
           (incf right-entropy (shannon-entropy right-probability))
           (incf right-entropy (shannon-entropy (- 1.0d0 right-probability)))))))
    (finally (return (values left-entropy right-entropy)))))


(defun total-entropy (target-data)
  (let* ((length (array-dimension target-data 0))
         (split-array (make-array length :element-type 'boolean
                                         :initial-element nil)))
    (nth-value 0 (split-entropy split-array target-data))))


(-> subsample-array ((simple-array double-float (* *))
                     fixnum (simple-array boolean (*))
                     boolean
                     fixnum)
    (simple-array double-float (* *)))
(defun subsample-array (array length split-array position skipped-column)
  (declare (optimize (speed 3) (safety 0)))
  (lret ((result (make-array `(,length ,(1- (array-dimension array 1)))
                             :element-type 'double-float)))
    (iterate
      (declare (type fixnum j i))
      (with j = 0)
      (for i from 0 below (array-dimension array 0))
      (when (eq position (aref split-array i))
        (iterate
          (declare (type fixnum k p))
          (with p = 0)
          (for k from 0 below (array-dimension array 1))
          (when (eql skipped-column k)
            (next-iteration))
          (setf (aref result j p) (aref array i k)
                p (1+ p)))
        (incf j)))))


(defun subsample-vector (vector skipped-position)
  (cl-ds.utils:copy-without vector skipped-position))


(defun make-simple-node (split-array shannon-entropy length
                         position parallel training-state
                         attribute-index)
  (if (not parallel)
      #1=(let ((new-state
                 (cl-grf.tp:training-state-clone
                  training-state
                  (subsample-array (cl-grf.tp:training-data training-state)
                                   length
                                   split-array position
                                   attribute-index)
                  (subsample-array (cl-grf.tp:target-data training-state)
                                   length
                                   split-array position
                                   attribute-index)
                  (~>  training-state cl-grf.tp:attribute-indexes
                       (subsample-vector attribute-index))))
               (new-leaf (make 'scored-leaf-node :score shannon-entropy
                                                 :support length)))
           (setf training-state nil split-array nil) ; so it can be gced
           (incf (cl-grf.tp:depth new-state))
           (let ((split-candidate (cl-grf.tp:split new-state new-leaf)))
             (if (null split-candidate)
                 new-leaf
                 split-candidate)))
      (lparallel:future #1#)))
