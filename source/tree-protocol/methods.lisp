(cl:in-package #:cl-grf.tree-protocol)


(defmethod (setf training-parameters) :round (new-value state)
  (check-type new-value fundamental-tree-training-parameters)
  (call-next-method new-value state))


(defmethod make-node (node-class &rest arguments)
  (apply #'make node-class arguments))


(defmethod (setf maximal-depth) :before (new-value
                                         training-parameters)
  (check-type new-value (integer 1 *)))


(defmethod (setf depth) :before (new-value training-parameters)
  (check-type new-value (integer 0 *)))


(defmethod (setf training-data) :before (new-value training-state)
  (cl-grf.data:check-data-points new-value))


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
    (declare (type cl-grf.data:data-matrix training-data)
             (type (integer 1 *) minimal-size))
    (if (or (< (cl-grf.data:data-points-count training-data)
               (* 2 minimal-size))
            (emptyp attribute-indexes)
            (>= depth maximal-depth))
        nil
        (call-next-method))))


(defun leaf-for (node data index)
  (declare (type cl-grf.data:data-matrix data)
           (type fixnum index))
  (if (typep node 'fundamental-leaf-node)
      node
      (bind ((attribute-index (attribute node))
             (attribute-value (attribute-value node)))
        (if (> (cl-grf.data:mref data index attribute-index)
               attribute-value)
            (leaf-for (right-node node) data index)
            (leaf-for (left-node node) data index)))))


(defmethod cl-grf.mp:predict ((model fundamental-tree-node) data
                              &optional parallel)
  (declare (ignore parallel))
  ;; TODO should be able to work in parallel
  (cl-grf.data:check-data-points data)
  (cl-grf.data:bind-data-matrix-dimensions
      ((data-points attributes data))
    (iterate
      (with result = nil)
      (with slice = (cl-grf.data:make-data-matrix 1 attributes))
      (for i from 0 below data-points)
      (iterate
        (for j from 0 below attributes)
        (setf (cl-grf.data:mref slice 0 j) (cl-grf.data:mref data i j)))
      (for leaf = (leaf-for model slice 0))
      (for prediction = (cl-grf.mp:predict leaf slice))
      (when (null result)
        (setf result (~>> (cl-grf.data:attributes-count prediction)
                          (cl-grf.data:make-data-matrix data-points))))
      (iterate
        (for j from 0 below attributes)
        (setf (cl-grf.data:mref slice i j) (cl-grf.data:mref prediction 0 j)))
      (finally (return result)))))


(defmethod shared-initialize :after
    ((instance fundamental-tree-training-parameters)
     slot-names
     &rest initargs)
  (declare (ignore slot-names initargs))
  (let ((maximal-depth (maximal-depth instance))
        (minimal-size (minimal-size instance))
        (trials-count (trials-count instance)))
    (parallel instance) ; here just to check if slot is bound
    (unless (integerp maximal-depth)
      (error 'type-error :expected 'integer
                         :datum maximal-depth))
    (unless (< 0 maximal-depth)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :maximal-depth
             :bounds '(< 0 :maximal-depth)
             :value maximal-depth))
    (unless (integerp minimal-size)
      (error 'type-error :expected 'integer
                         :datum minimal-size))
    (unless (<= 0 minimal-size)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :minimal-size
             :bounds '(<= 0 :minimal-size)
             :value minimal-size))
    (unless (integerp trials-count)
      (error 'type-error :expected 'integer
                         :datum trials-count))
    (unless (< 0 trials-count)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :trials-count
             :bounds '(< 0 :trials-count)
             :value trials-count))))
