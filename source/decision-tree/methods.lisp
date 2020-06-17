(cl:in-package #:statistical-learning.decision-tree)


(defmethod sl.tp:calculate-loss* ((parameters fundamental-decision-tree-parameters)
                                  state
                                  split-array)
  (sl.opt:loss (optimized-function parameters)
               (sl.tp:target-data state)
               (sl.tp:weights state)
               split-array))


(defmethod sl.tp:make-training-state ((parameters fundamental-decision-tree-parameters)
                                      train-data
                                      target-data
                                      &rest initargs
                                      &key weights attributes &allow-other-keys)
  (declare (ignore initargs))
  (let ((optimized-function (optimized-function parameters)))
    (make 'sl.tp:fundamental-training-state
          :training-parameters parameters
          :loss (sl.opt:loss optimized-function
                             target-data
                             weights)
          :weights weights
          :attribute-indexes attributes
          :target-data target-data
          :training-data train-data)))


(defmethod sl.mp:make-model ((parameters fundamental-decision-tree-parameters)
                             train-data
                             target-data
                             &key weights attributes
                               (state (sl.tp:make-training-state parameters
                                                                 train-data
                                                                 target-data
                                                                 :weights weights
                                                                 :attributes attributes)))
  (make 'sl.tp:tree-model
        :parameters parameters
        :root (~>> state sl.tp:make-leaf (sl.tp:split state))))


(defmethod sl.tp:initialize-leaf ((training-parameters classification)
                                  training-state
                                  leaf)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((target-data (sl.tp:target-data training-state))
         (number-of-classes (~> training-parameters
                                optimized-function
                                sl.opt:number-of-classes))
         (data-points-count (sl.data:data-points-count target-data))
         (score (sl.tp:loss training-state))
         (predictions (sl.data:make-data-matrix 1 number-of-classes)))
    (declare (type fixnum number-of-classes data-points-count)
             (type statistical-learning.data:data-matrix target-data predictions))
    (iterate
      (declare (type fixnum i index))
      (for i from 0 below data-points-count)
      (for index = (truncate (sl.data:mref target-data i 0)))
      (incf (sl.data:mref predictions 0 index)))
    (setf (sl.tp:support leaf) data-points-count
          (sl.tp:predictions leaf) predictions
          (sl.tp:loss leaf) score)))


(defmethod sl.tp:initialize-leaf ((training-parameters regression)
                                  training-state
                                  leaf)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((target-data (sl.tp:target-data training-state))
         (score (sl.tp:loss training-state))
         (sum 0.0d0)
         (data-points-count (sl.data:data-points-count target-data)))
    (declare (type fixnum data-points-count)
             (type double-float sum))
    (iterate
      (declare (type fixnum i)
               (type double-float sum))
      (for i from 0 below data-points-count)
      (incf sum (sl.data:mref target-data i 0)))
    (setf (sl.tp:support leaf) data-points-count
          (sl.tp:predictions leaf) (/ sum data-points-count)
          (sl.tp:loss leaf) score)))


(defmethod sl.tp:contribute-predictions* ((parameters regression)
                                          model
                                          data
                                          state
                                          parallel)
  (sl.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (when (null state)
      (setf state (make 'sl.tp:contributed-predictions
                        :indexes (sl.data:iota-vector data-points-count)
                        :training-parameters parameters
                        :sums (sl.data:make-data-matrix data-points-count
                                                        1))))
    (let* ((sums (sl.tp:sums state))
           (root (sl.tp:root model)))
      (funcall (if parallel #'lparallel:pmap #'map)
               nil
               (lambda (data-point)
                 (let* ((leaf (sl.tp:leaf-for root data data-point))
                        (predictions (sl.tp:predictions leaf)))
                   (incf (sl.data:mref sums data-point 0)
                         predictions)))
               (sl.tp:indexes state)))
    (incf (sl.tp:contributions-count state))
    state))


(defmethod sl.tp:contribute-predictions* ((parameters classification)
                                          model
                                          data
                                          state
                                          parallel)
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
             (root (sl.tp:root model)))
        (funcall (if parallel #'lparallel:pmap #'map)
                 nil
                 (lambda (data-point)
                   (iterate
                     (declare (type fixnum j))
                     (with leaf = (sl.tp:leaf-for root data data-point))
                     (with predictions = (sl.tp:predictions leaf))
                     (with support = (sl.tp:support leaf))
                     (for j from 0 below number-of-classes)
                     (for class-support = (sl.data:mref predictions 0 j))
                     (incf (sl.data:mref sums data-point j)
                           (/ class-support support))))
                 (sl.tp:indexes state))))
    (incf (sl.tp:contributions-count state))
    state))


(defmethod sl.tp:extract-predictions* ((parameters fundamental-decision-tree-parameters)
                                       (state sl.tp:contributed-predictions))
  (let ((count (sl.tp:contributions-count state)))
    (sl.data:map-data-matrix (lambda (value) (/ value count))
                             (sl.tp:sums state))))


(defmethod sl.opt:number-of-classes ((object classification))
  (~> object optimized-function sl.opt:number-of-classes))
