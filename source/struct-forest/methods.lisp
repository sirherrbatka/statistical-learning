(cl:in-package #:statistical-learning.struct-forest)


(defmethod sl.mp:target-data ((state struct-state))
  (ensure (relabaled state)
    (relable (~> state sl.mp:training-parameters original relabeler)
             (~> state sl.mp:training-parameters original)
             state)))


(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters struct)
                                            &rest initargs
                                            &key weights)
  (lret ((result
          (apply #'make 'struct-state
                 :training-parameters (training-implementation parameters)
                 :loss 0.0d0
                 initargs)))
    (setf (relabaled result) (relable (relabeler parameters)
                                      parameters
                                      result))
    (setf (sl.tp:loss result) (sl.opt:loss (sl.opt:optimized-function (sl.mp:training-parameters result))
                                           (relabaled result)
                                           weights))))


(defmethod sl.tp:initialize-leaf/proxy (parameters/proxy
                                        (training-parameters struct-training-implementation)
                                        training-state
                                        leaf)
  (let* ((target-data (struct-target-data training-state))
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
     (parameters struct)
     model
     data
     state
     context
     parallel
     &optional (leaf-key #'identity))
  (sl.tp:contribute-predictions*/proxy parameters/proxy
                                       (prediction-implementation parameters)
                                       model
                                       data
                                       state
                                       context
                                       parallel
                                       leaf-key))


(defmethod struct-target-data ((state struct-state))
  (slot-value state 'sl.tp::%target-data))


(defmethod training-implementation
    ((parameters struct) &rest initargs)
  (apply #'make 'struct-training-implementation
         :proxy (sl.common:proxy parameters)
         :original parameters
         :splitter (sl.tp:splitter parameters)
         :maximal-depth (sl.tp:maximal-depth parameters)
         :minimal-difference (sl.tp:minimal-difference parameters)
         :minimal-size (sl.tp:minimal-size parameters)
         :parallel (sl.tp:parallel parameters)
         initargs))


(defmethod prediction-implementation ((parameters struct)
                                       &rest initargs)
  (apply #'make 'sl.dt:regression
         :proxy (sl.common:proxy parameters)
         :splitter (sl.tp:splitter parameters)
         initargs))


(defmethod sl.tp:split-training-state*/proxy
    (parameters/proxy
     (parameters struct-training-implementation)
     state split-array
     position size initargs
     point)
  (bind ((result (call-next-method)))
    (setf (sl.mp:target-data result)
          (sl.data:data-matrix-quasi-clone (struct-target-data state)
                                           :index (~> result sl.mp:train-data sl.data:index)))
    result))


(defmethod relabel-iterations ((parameters struct-training-implementation))
  (~> parameters original relabel-iterations))


(defmethod relabel-repeats ((parameters struct-training-implementation))
  (~> parameters original relabel-repeats))


(defmethod relable ((relabler euclid-distance-relabaler) parameters state)
  (bind (((:values first second) (select-pivots parameters state)))
    (declare (type (simple-array fixnum (*)) data-points)
             (type sl.data:double-float-data-matrix target-data)
             (type fixnum first second))
    (relabel-with-pivots state first second)))
