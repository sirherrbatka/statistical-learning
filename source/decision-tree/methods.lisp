(cl:in-package #:statistical-learning.decision-tree)


(defmethod sl.tp:calculate-loss*/proxy
    (parameters/proxy
     (parameters fundamental-decision-tree-parameters)
     state
     split-array
     left-length
     right-length)
  (sl.opt:loss (optimized-function parameters)
               (sl.mp:target-data state)
               (sl.mp:weights state)
               split-array))


(defmethod sl.mp:make-training-state/proxy
    (parameters/proxy
     (parameters fundamental-decision-tree-parameters)
     &rest initargs
     &key train-data
       target-data weights attributes &allow-other-keys)
  (declare (ignore initargs))
  (let ((optimized-function (optimized-function parameters)))
    (make 'sl.tp:tree-training-state
          :training-parameters parameters
          :loss (sl.opt:loss optimized-function target-data weights)
          :weights weights
          :attributes attributes
          :target-data target-data
          :train-data train-data)))


(defmethod sl.tp:initialize-leaf/proxy
    (parameters/proxy
     (training-parameters classification)
     training-state
     leaf)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((target-data (sl.mp:target-data training-state))
         (number-of-classes (~> training-parameters
                                optimized-function
                                sl.opt:number-of-classes))
         (data-points-count (sl.data:data-points-count target-data))
         (predictions (make-array `(1 ,number-of-classes)
                                  :element-type 'single-float
                                  :initial-element 0.0)))
    (declare (type fixnum number-of-classes data-points-count)
             (type sl.data:single-float-data-matrix target-data))
    (iterate
      (declare (type fixnum i index))
      (for i from 0 below data-points-count)
      (for index = (truncate (sl.data:mref target-data i 0)))
      (incf (aref predictions 0 index)))
    (setf (sl.tp:predictions leaf)
          (sl.data:array-avg predictions data-points-count))))


(defmethod sl.tp:initialize-leaf/proxy
    (parameters/proxy
     (training-parameters regression)
     training-state
     leaf)
  (let* ((target-data (sl.mp:target-data training-state))
         (attributes-count (sl.data:attributes-count target-data))
         (result (make-array `(1 ,attributes-count) :element-type 'single-float :initial-element 0.0))
         (data-points-count (sl.data:data-points-count target-data)))
    (declare (type fixnum data-points-count))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below data-points-count)
      (iterate
        (declare (type fixnum ii))
        (for ii from 0 below attributes-count)
        (incf (aref result 0 ii) (sl.data:mref target-data i ii))))
    (setf (sl.tp:predictions leaf)
          (sl.data:array-avg result data-points-count))))


(defmethod sl.tp:contribute-predictions*/proxy
    (parameters/proxy
     (parameters regression)
     model
     data
     state
     context
     parallel
     &optional (leaf-key #'identity))
  (ensure leaf-key #'identity)
  (sl.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (when (null state)
      (setf state (make 'sl.tp:contributed-predictions
                        :sums nil
                        :training-parameters parameters)))
    (let* ((splitter (sl.tp:splitter parameters))
           (lock (bt:make-lock))
           (weight (sl.tp:weight model))
           (root (sl.tp:root model)))
      (sl.data:data-matrix-map-data-points
       (lambda (data-point data)
         (let* ((leafs (sl.tp:leaf-for splitter root
                                       data data-point
                                       model))
                (sums (bt:with-lock-held (lock)
                        (ensure (sl.tp:sums state)
                          (make-array (list data-points-count attributes-count)
                                    :element-type 'single-float)))))
           (if (vectorp leafs)
               (iterate
                 (declare (type fixnum i))
                 (with length = (length leafs))
                 (for i from 0 below length)
                 (for leaf = (funcall leaf-key (aref leafs i)))
                 (for predictions = (sl.tp:predictions leaf))
                 (for attributes-count = (array-dimension predictions 1))
                 (iterate
                   (for i from 0 below attributes-count)
                   (incf (aref sums data-point i)
                         (/ (* weight (aref predictions 0 i)) length))))
               (let* ((predictions (sl.tp:predictions leafs))
                      (attributes-count (array-dimension predictions 1)))
                 (iterate
                   (declare (type fixnum i))
                   (for i from 0 below attributes-count)
                   (incf (aref sums data-point i)
                         (* weight (aref predictions 0 i))))))))
       data
       parallel)
      (incf (sl.tp:contributions-count state) weight)
      state)))


(defmethod sl.tp:contribute-predictions*/proxy
    (parameters/proxy
     (parameters classification)
     model
     data
     state
     context
     parallel
     &optional (leaf-key #'identity))
  (ensure leaf-key #'identity)
  (sl.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (let ((number-of-classes (~> parameters
                                 optimized-function
                                 sl.opt:number-of-classes))
          (weight (sl.tp:weight model)))
      (when (null state)
        (setf state (make 'sl.tp:contributed-predictions
                          :training-parameters parameters
                          :sums (make-array (list data-points-count number-of-classes)
                                            :element-type 'single-float))))
      (let* ((sums (sl.tp:sums state))
             (splitter (sl.tp:splitter parameters))
             (root (sl.tp:root model)))
        (sl.data:data-matrix-map-data-points
                 (lambda (data-point data &aux (leafs (sl.tp:leaf-for splitter root
                                                                 data data-point
                                                                 model)))
                   (if (vectorp leafs)
                       (iterate
                         (declare (type fixnum i))
                         (with length = (length leafs))
                         (for i from 0 below length)
                         (for l = (aref leafs i))
                         (for leaf = (funcall leaf-key l))
                         (for predictions = (sl.tp:predictions leaf))
                         (iterate
                           (declare (type fixnum j))
                           (for j from 0 below number-of-classes)
                           (for class-support = (aref predictions 0 j))
                           (incf (aref sums data-point j) (/ (* weight class-support) length))))
                       (iterate
                         (declare (type fixnum j))
                         (with leaf = (funcall leaf-key leafs))
                         (with predictions = (sl.tp:predictions leaf))
                         (for j from 0 below number-of-classes)
                         (for class-support = (aref predictions 0 j))
                         (incf (aref sums data-point j) (* weight class-support)))))
                 data
                 parallel))
      (incf (sl.tp:contributions-count state) weight))
    state))


(defmethod sl.tp:extract-predictions*/proxy
    (parameters/proxy
     (parameters fundamental-decision-tree-parameters)
     (state sl.tp:contributed-predictions))
  (let ((count (sl.tp:contributions-count state)))
    (sl.data:array-avg (sl.tp:sums state)
                       count)))


(defmethod sl.opt:number-of-classes ((object classification))
  (~> object optimized-function sl.opt:number-of-classes))


(defmethod cl-ds.utils:cloning-information
    append ((object fundamental-decision-tree-parameters))
  '((:optimized-function optimized-function)))
