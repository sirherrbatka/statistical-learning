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


(-> random-test (cl-grf.data:data-matrix)
    (values fixnum double-float))
(defun random-test (data)
  (declare (optimize (speed 3) (safety 0)))
  (bind ((attribute (random (array-dimension data 1)))
         ((:values min max) (data-min/max data attribute))
         (threshold (if (= (the double-float min)
                           (the double-float max))
                        min
                        (random-uniform min max))))
    (values attribute threshold)))


(-> fill-split-array (cl-grf.data:data-matrix
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
    (for check = (> (cl-grf.data:mref data i attribute) threshold))
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
                   cl-grf.data:data-matrix)
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
            (incf right-sum (cl-grf.data:mref target-data i j))
            (incf right-count))
          (progn
            (incf left-sum (cl-grf.data:mref target-data i j))
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
  (let* ((length (cl-grf.data:data-points-count target-data))
         (split-array (make-array length :element-type 'boolean
                                         :initial-element nil)))
    (nth-value 0 (split-entropy split-array target-data))))


(-> subsample-array (cl-grf.data:data-matrix
                     fixnum (simple-array boolean (*))
                     boolean
                     (or null skipped-column))
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
                    p (1+ p)))
            (incf j)))))))


(defun subsample-vector (vector skipped-position)
  (cl-ds.utils:copy-without vector skipped-position))


(defun make-simple-node (split-array shannon-entropy length
                         position parallel training-state
                         attribute-index)
  (if (not parallel)
      #1=(let* ((new-state
                  (cl-grf.tp:training-state-clone
                   training-state
                   (subsample-array (cl-grf.tp:training-data training-state)
                                    length
                                    split-array position
                                    attribute-index)
                   (subsample-array (cl-grf.tp:target-data training-state)
                                    length
                                    split-array position
                                    nil)
                   (~>  training-state cl-grf.tp:attribute-indexes
                        (subsample-vector attribute-index))))
                (target-data (cl-grf.tp:target-data new-state))
                (predictions (~>> target-data cl-grf.data:attributes-count
                                  (cl-grf.data:make-data-matrix 1)))
                (new-leaf (make 'scored-leaf-node :score shannon-entropy
                                                  :predictions predictions
                                                  :support length)))
           (cl-grf.data:bind-data-matrix-dimensions
               ((data-points-count attributes-count target-data))
             (iterate
               (declare (type fixnum i))
               (for i from 0 below data-points-count)
               (iterate
                 (for j from 0 below attributes-count)
                 (declare (type fixnum j))
                 (incf (cl-grf.data:mref predictions 0 j)
                       (cl-grf.data:mref target-data i j)))))
           (setf training-state nil
                 split-array nil) ; so it can be gced
           (incf (cl-grf.tp:depth new-state))
           (let ((split-candidate (cl-grf.tp:split new-state new-leaf)))
             (if (null split-candidate)
                 new-leaf
                 split-candidate)))
      (lparallel:future #1#)))
