(cl:in-package #:statistical-learning.decision-tree)


(defmethod sl.tp:calculate-loss*/proxy
    (parameters/proxy
     (parameters fundamental-decision-tree-parameters)
     state
     split-array
     left-length
     right-length
     middle-length)
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
                                  :element-type 'double-float
                                  :initial-element 0.0d0)))
    (declare (type fixnum number-of-classes data-points-count)
             (type sl.data:double-float-data-matrix target-data))
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
         (result (make-array `(1 ,attributes-count) :element-type 'double-float :initial-element 0.0d0))
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
      (sl.data:data-matrix-map
       (lambda (data-point data)
         (let* ((leaf (~>> (sl.tp:leaf-for splitter root
                                           data data-point
                                           model)
                           (funcall leaf-key)))
                (predictions (sl.tp:predictions leaf))
                (attributes-count (array-dimension predictions 1))
                (sums (bt:with-lock-held (lock)
                        (ensure (sl.tp:sums state)
                          (sl.data:make-data-matrix data-points-count
                                                    attributes-count)))))
           (iterate
             (for i from 0 below attributes-count)
             (incf (sl.data:mref sums data-point i)
                   (* weight (aref predictions 0 i))))))
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
                          :sums (sl.data:make-data-matrix data-points-count
                                                          number-of-classes))))
      (let* ((sums (sl.tp:sums state))
             (splitter (sl.tp:splitter parameters))
             (root (sl.tp:root model)))
        (sl.data:data-matrix-map
                 (lambda (data-point data)
                   (iterate
                     (declare (type fixnum j))
                     (with leaf = (~>> (sl.tp:leaf-for splitter root
                                                       data data-point
                                                       model)
                                       (funcall leaf-key)))
                     (with predictions = (sl.tp:predictions leaf))
                     (for j from 0 below number-of-classes)
                     (for class-support = (aref predictions 0 j))
                     (incf (sl.data:mref sums data-point j) (* weight class-support))))
                 data
                 parallel))
      (incf (sl.tp:contributions-count state) weight))
    state))


(defmethod sl.tp:extract-predictions*/proxy
    (parameters/proxy
     (parameters fundamental-decision-tree-parameters)
     (state sl.tp:contributed-predictions))
  (let ((count (sl.tp:contributions-count state)))
    (sl.data:data-matrix-map (lambda (data-point data)
                               (iterate
                                 (for i from 0 below (array-dimension data 1))
                                 (setf #1=(aref data data-point i) (/ #1# count))))
                             (sl.tp:sums state)
                             nil)))


(defmethod sl.opt:number-of-classes ((object classification))
  (~> object optimized-function sl.opt:number-of-classes))


(defmethod cl-ds.utils:cloning-information
    append ((object fundamental-decision-tree-parameters))
  '((:optimized-function optimized-function)))
