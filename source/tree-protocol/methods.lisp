(cl:in-package #:statistical-learning.tree-protocol)


(defmethod (setf training-parameters) :around (new-value state)
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
  (statistical-learning.data:check-data-points new-value))


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
    ((object tree-training-state))
  `((:depth depth)
    (:loss loss)
    (:attribute-indexes attribute-indexes)))


(defmethod split* :around ((training-parameters fundamental-tree-training-parameters)
                           training-state leaf)
  (let* ((training-data (sl.mp:training-data training-state))
         (depth (depth training-state))
         (attribute-indexes (attribute-indexes training-state))
         (loss (loss leaf))
         (maximal-depth (maximal-depth training-parameters))
         (minimal-size (minimal-size training-parameters)))
    (declare (type statistical-learning.data:data-matrix training-data)
             (type (integer 1 *) minimal-size))
    (if (or (< (statistical-learning.data:data-points-count training-data)
               (* 2 minimal-size))
            (emptyp attribute-indexes)
            (>= depth maximal-depth)
            (<= loss (minimal-difference training-parameters)))
        nil
        (call-next-method))))


(defmethod statistical-learning.mp:predict ((model tree-model)
                                            data
                                            &optional parallel)
  (~> (contribute-predictions model data nil parallel)
      extract-predictions))


(defmethod initialize-instance :after
    ((instance fundamental-tree-training-parameters)
     &rest initargs)
  (declare (ignore initargs))
  (let ((maximal-depth (maximal-depth instance))
        (minimal-size (minimal-size instance))
        (minimal-difference (minimal-difference instance))
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
