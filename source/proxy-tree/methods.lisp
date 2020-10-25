(cl:in-package #:sl.proxy-tree)


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


(defmethod sl.mp:data-points ((state proxy-state))
  (~> state inner sl.mp:data-points))


(defmethod (setf sl.mp:data-points) (new-value (state proxy-state))
  (setf (sl.mp:data-points (inner state)) new-value))
