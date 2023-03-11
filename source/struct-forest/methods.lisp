(cl:in-package #:statistical-learning.struct-forest)


(defmethod sl.mp:target-data ((state struct-state))
  (ensure (relabaled state)
    (relable (~> state sl.mp:parameters original)
             state)))


(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters struct-training-implementation)
                                            &rest initargs)
  (declare (ignore initargs))
  (lret ((result (call-next-method)))
    (change-class result 'struct-state)))


(defmethod sl.tp:initialize-leaf/proxy (parameters/proxy
                                        (training-parameters struct-training-implementation)
                                        training-state
                                        leaf)
  cl-ds.utils:todo)


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


(defmethod relable ((parameters struct) state)
  cl-ds.utils:todo)
