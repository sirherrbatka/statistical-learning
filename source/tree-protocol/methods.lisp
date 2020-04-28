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
  (check-type new-value (simple-array double-float (* *)))
  (call-next-method new-value training-state))


(defmethod (setf minimal-size) :around (new-value
                                        training-parameters)
  (check-type new-value positive-integer)
  (call-next-method new-value training-parameters))


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


(defmethod cl-ds.utils:cloning-information append
    ((object fundamental-training-state))
  `((:training-parameters training-parameters)
    (:depth depth)
    (:attribute-indexes attribute-indexes)
    (:target-data target-data)
    (:training-data training-data)))


(defmethod split* :around (training-parameters training-state leaf)
  (let* ((training-data (training-data training-state))
         (depth (depth training-state))
         (attribute-indexes (attribute-indexes training-state))
         (maximal-depth (maximal-depth training-parameters))
         (minimal-size (minimal-size training-parameters)))
    (declare (type (simple-array double-float (* *)) training-data)
             (type (integer 1 *) minimal-size))
    (if (or (< (array-dimension training-data 0) minimal-size)
            (or (emptyp attribute-indexes))
            (= depth maximal-depth))
        nil
        (call-next-method))))


(defmethod split ((split fundamental-split-candidate))
  (make-class 'fundamental-tree-node
              :left-node (left-node split)
              :right-node (right-node split)
              :attribute (attribute split)))


(defmethod leaf-for ((node fundamental-leaf-node) data index)
  node)


(defmethod leaf-for ((node fundamental-tree-node) data index)
  (declare (type (simple-array double-float (* *)) data)
           (type fixnum index))
  (let* ((attribute-index (attribute node))
         (attribute-value (attribute-value node))
         (right-p (> (aref data index attribute-index)
                     attribute-value)))
    (declare (type fixnum attribute-index)
             (type double-float attribute-value))
    (if right-p
        (leaf-for (right-node node) data index)
        (leaf-for (left-node node) data index))))
