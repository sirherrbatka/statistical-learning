(cl:in-package #:sl.proxy-tree)


(define-forwarding
  (sl.tp:extract-predictions* (predictions))
  (sl.tp:make-leaf* ())
  (sl.tp:splitter ())
  (sl.tp:calculate-loss* (state split-array))
  (sl.tp:initialize-leaf (state leaf))
  (sl.tp:maximal-depth ())
  (sl.tp:minimal-size ())
  (sl.tp:minimal-difference ())
  (sl.perf:errors (target predictions))
  (sl.tp:trials-count ())
  (sl.perf:average-performance-metric (metrics))
  (sl.tp:parallel ()))


(defmethod sl.tp:contribute-predictions* ((training-parameters proxy-tree)
                                          model
                                          data
                                          state
                                          parallel
                                          &optional leaf-key)
  (if (null leaf-key)
      (sl.tp:contribute-predictions* (inner training-parameters)
                                     model
                                     data
                                     state
                                     parallel)
      (sl.tp:contribute-predictions* (inner training-parameters)
                                     model
                                     data
                                     state
                                     parallel
                                     leaf-key)))


(defmethod sl.perf:performance-metric* ((parameters proxy-tree)
                                        type
                                        target
                                        predictions
                                        weights)
  (sl.perf:performance-metric* (inner parameters)
                               type
                               target
                               predictions
                               weights))


(defmethod cl-ds.utils:cloning-information append ((state proxy-state))
  `((:inner inner)))


(defmethod sl.mp:train-data ((state proxy-state))
  (sl.mp:train-data (inner state)))


(defmethod sl.mp:target-data ((state proxy-state))
  (sl.mp:target-data (inner state)))


(defmethod sl.mp:weights ((state proxy-state))
  (sl.mp:weights (inner state)))


(defmethod sl.tp:attribute-indexes ((state proxy-state))
  (sl.tp:attribute-indexes (inner state)))


(defmethod sl.tp:depth ((state proxy-state))
  (sl.tp:depth (inner state)))


(defmethod sl.tp:loss ((state proxy-state))
  (sl.tp:loss (inner state)))


(defmethod sl.mp:make-model* ((parameters proxy-tree) training-state)
  (call-next-method))


(defmethod inner ((parameters sl.tp:fundamental-tree-training-parameters))
  parameters)


(defmethod sl.mp:data-points ((state proxy-state))
  (~> state inner sl.mp:data-points))


(defmethod (setf sl.mp:data-points) (new-value (state proxy-state))
  (setf (sl.mp:data-points (inner state)) new-value))


(defmethod sl.tp:requires-split-p and ((splitter sl.tp:fundamental-splitter)
                                       (training-parameters proxy-tree)
                                       training-state)
  (let ((inner-state (inner training-state))
        (inner-parameters (inner training-parameters)))
    (sl.tp:requires-split-p (sl.tp:splitter inner-parameters)
                            inner-parameters
                            inner-state)))
