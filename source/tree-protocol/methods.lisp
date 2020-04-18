(cl:in-package #:cl-grf.tree-protocol)


(defmethod (setf training-parameters) :round (new-value state)
  (check-type new-value fundamental-training-parameters)
  (call-next-method new-value state))


(defmethod make-node (node-class &rest arguments)
  (apply #'make node-class arguments))


(defmethod (setf maximal-depth) :around (new-value
                                         training-parameters)
  (check-type new-value (integer 1 *))
  (call-next-method new-value training-parameters))


(defmethod (setf depth) :around (new-value training-parameters)
  (check-type new-value (integer 0 *))
  (call-next-method new-value training-parameters))


(defmethod (setf training-data) :around (new-value training-state)
  (check-type new-value simple-vector)
  (call-next-method new-value training-state))


(defmethod treep ((node fundamental-tree-node))
  t)


(defmethod treep ((node fundamental-leaf-node))
  nil)


(defmethod force-tree* ((node fundamental-tree-node))
  (setf (left-node node) (~> node left-node lparallel:force)
        (right-node node) (~> node right-node lparallel:force))
  (unless (null (left-node node))
    (~> node left-node force-tree*))
  (unless (null (right-node node))
    (~> node right-node force-tree*))
  node)


(defmethod force-tree* ((node fundamental-leaf-node))
  node)


(defmethod cl-ds.utils:cloning-information :append
    ((object fundamental-training-state))
  `((:training-parameters training-parameters)
    (:split-mode split-mode)
    (:needs-split-p-mode needs-split-p-mode)
    (:depth depth)
    (:training-data training-data)))
