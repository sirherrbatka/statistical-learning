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
  "Uses ExtraTree approach."
  (declare (optimize (speed 3) (safety 0)))
  (bind ((attribute (~> data cl-grf.data:attributes-count random))
         ((:values min max) (data-min/max data attribute))
         (threshold (random-uniform min max)))
    (values attribute threshold)))


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


(-> vector-entropy ((simple-array double-float (*))) double-float)
(defun vector-entropy (sums)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (let ((grand-total (reduce #'+ sums :initial-value 0.0d0)))
    (declare (type double-float grand-total))
    (if (zerop grand-total)
        0.0d0
        (iterate
          (declare (type fixnum length i)
                   (type double-float entropy p))
          (with entropy = 0.0d0)
          (with length = (length sums))
          (for i from 0 below length)
          (for sum = (aref sums i))
          (when (zerop sum)
            (next-iteration))
          (for p = (/ sum grand-total))
          (decf entropy (* p (the double-float (log p))))
          (finally (return entropy))))))


(-> split-entropy (single-information-gain-classification
                   (simple-array boolean (*))
                   cl-grf.data:data-matrix)
    (values double-float double-float))
(defun split-entropy (parameters split-array target-data)
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
        (incf (aref right-sums target))
        (incf (aref left-sums target)))
    (finally (return (values (vector-entropy left-sums)
                             (vector-entropy right-sums))))))


(defun total-entropy (parameters target-data)
  (let* ((length (cl-grf.data:data-points-count target-data))
         (split-array (make-array length :element-type 'boolean
                                         :initial-element nil)))
    (nth-value 0 (split-entropy parameters split-array target-data))))


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
                    p (1+ p)))
            (incf j)))))))


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
                               attribute-index))


(defun make-simple-node (split-array shannon-entropy length
                         position parallel training-state
                         attribute-index)
  (make-simple-node* (cl-grf.tp:training-parameters training-state)
                     split-array
                     shannon-entropy
                     length
                     position
                     parallel
                     training-state
                     attribute-index))


(defmethod make-simple-node*
    ((parameters single-information-gain-classification)
     split-array
     shannon-entropy
     length
     position
     parallel
     training-state
     attribute-index)
  (if (not parallel)
      #1=(let* ((number-of-classes (number-of-classes parameters))
                (old-target-data (cl-grf.tp:target-data training-state))
                (new-attributes (~>  training-state
                                     cl-grf.tp:attribute-indexes
                                     (subsample-vector attribute-index)))
                (new-state
                  (cl-grf.tp:training-state-clone
                   training-state
                   (subsample-array (cl-grf.tp:training-data training-state)
                                    length split-array
                                    position attribute-index)
                   (subsample-array old-target-data
                                    length split-array
                                    position nil)
                   new-attributes))
                (predictions (cl-grf.data:make-data-matrix
                              1 number-of-classes))
                (new-leaf (make 'scored-leaf-node
                                :score shannon-entropy
                                :predictions predictions
                                :support length)))
           (iterate
             (declare (type fixnum i))
             (for i from 0 below (length split-array))
             (unless (eq position (aref split-array i))
               (next-iteration))
             (for prediction = (truncate (cl-grf.data:mref old-target-data
                                                           i 0)))
             (incf (cl-grf.data:mref predictions 0 prediction)
                   1.0d0))
           (setf training-state nil
                 old-target-data nil
                 split-array nil) ; so it can be gced
           (incf (cl-grf.tp:depth new-state))
           (let ((split-candidate (cl-grf.tp:split new-state new-leaf)))
             (if (null split-candidate)
                 new-leaf
                 split-candidate)))
      (lparallel:future #1#)))
