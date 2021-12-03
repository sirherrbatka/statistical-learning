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
               (sl.mp:data-points state)
               split-array))


(defmethod sl.mp:make-training-state/proxy
    (parameters/proxy
     (parameters fundamental-decision-tree-parameters)
     &rest initargs
     &key train-data data-points
       target-data weights attributes &allow-other-keys)
  (declare (ignore initargs))
  (let ((optimized-function (optimized-function parameters)))
    (make 'sl.tp:tree-training-state
          :training-parameters parameters
          :data-points data-points
          :loss (sl.opt:loss optimized-function target-data weights data-points)
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
         (data-points (sl.mp:data-points training-state))
         (number-of-classes (~> training-parameters
                                optimized-function
                                sl.opt:number-of-classes))
         (data-points-count (length data-points))
         (predictions (sl.data:make-data-matrix 1 number-of-classes)))
    (declare (type fixnum number-of-classes data-points-count)
             (type (simple-array fixnum (*)) data-points)
             (type sl.data:double-float-data-matrix target-data predictions))
    (iterate
      (declare (type fixnum j i index))
      (for j from 0 below data-points-count)
      (for i = (aref data-points j))
      (for index = (truncate (sl.data:mref target-data i 0)))
      (incf (sl.data:mref predictions 0 index)))
    (setf (sl.tp:predictions leaf)
          (sl.data:data-matrix-avg predictions data-points-count))))


(defmethod sl.tp:initialize-leaf/proxy
    (parameters/proxy
     (training-parameters regression)
     training-state
     leaf)
  (let* ((target-data (sl.mp:target-data training-state))
         (attributes-count (sl.data:attributes-count target-data))
         (result (sl.data:make-data-matrix 1 attributes-count))
         (data-points (sl.mp:data-points training-state))
         (data-points-count (length data-points)))
    (declare (type fixnum data-points-count)
             (type (simple-array fixnum (*)) data-points))
    (iterate
      (declare (type fixnum j i))
      (for j from 0 below data-points-count)
      (for i = (aref data-points j))
      (iterate
        (declare (type fixnum ii))
        (for ii from 0 below attributes-count)
        (incf (sl.data:mref result 0 ii) (sl.data:mref target-data i ii))))
    (setf (sl.tp:predictions leaf)
          (sl.data:data-matrix-avg result data-points-count))))


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
                        :indexes (sl.data:iota-vector data-points-count)
                        :sums nil
                        :training-parameters parameters)))
    (let* ((splitter (sl.tp:splitter parameters))
           (lock (bt:make-lock))
           (root (sl.tp:root model)))
      (funcall (if parallel #'lparallel:pmap #'map)
               nil
               (lambda (data-point)
                 (let* ((leaf (~>> (sl.tp:leaf-for splitter root
                                                   data data-point
                                                   model)
                                   (funcall leaf-key)))
                        (predictions (sl.tp:predictions leaf))
                        (attributes-count (sl.data:attributes-count predictions))
                        (sums (bt:with-lock-held (lock)
                                (ensure (sl.tp:sums state)
                                  (sl.data:make-data-matrix data-points-count
                                                            attributes-count)))))
                   (iterate
                     (for i from 0 below attributes-count)
                     (incf (sl.data:mref sums data-point i)
                           (sl.data:mref predictions 0 i)))))
               (sl.tp:indexes state)))
    (incf (sl.tp:contributions-count state))
    state))


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
                                 sl.opt:number-of-classes)))
      (when (null state)
        (setf state (make 'sl.tp:contributed-predictions
                          :indexes (sl.data:iota-vector data-points-count)
                          :training-parameters parameters
                          :sums (sl.data:make-data-matrix data-points-count
                                                          number-of-classes))))
      (let* ((sums (sl.tp:sums state))
             (splitter (sl.tp:splitter parameters))
             (root (sl.tp:root model)))
        (funcall (if parallel #'lparallel:pmap #'map)
                 nil
                 (lambda (data-point)
                   (iterate
                     (declare (type fixnum j))
                     (with leaf = (~>> (sl.tp:leaf-for splitter root
                                                       data data-point
                                                       model)
                                       (funcall leaf-key)))
                     (with predictions = (sl.tp:predictions leaf))
                     (for j from 0 below number-of-classes)
                     (for class-support = (sl.data:mref predictions 0 j))
                     (incf (sl.data:mref sums data-point j) class-support)))
                 (sl.tp:indexes state))))
    (incf (sl.tp:contributions-count state))
    state))


(defmethod sl.tp:extract-predictions*/proxy
    (parameters/proxy
     (parameters fundamental-decision-tree-parameters)
     (state sl.tp:contributed-predictions))
  (let ((count (sl.tp:contributions-count state)))
    (sl.data:map-data-matrix (lambda (value) (/ value count))
                             (sl.tp:sums state))))


(defmethod sl.opt:number-of-classes ((object classification))
  (~> object optimized-function sl.opt:number-of-classes))


(defmethod cl-ds.utils:cloning-information
    append ((object fundamental-decision-tree-parameters))
  '((:optimized-function optimized-function)))
