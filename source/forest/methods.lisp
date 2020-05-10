(cl:in-package #:cl-grf.forest)


(defmethod shared-initialize :after
    ((instance random-forest-parameters)
     slot-names
     &rest initargs)
  (declare (ignore initargs slot-names))
  (let* ((trees-count (trees-count instance))
         (forest-class (forest-class instance))
         (parallel (parallel instance))
         (tree-attributes-count (tree-attributes-count instance))
         (tree-sample-size (tree-sample-size instance))
         (tree-parameters (tree-parameters instance)))
    (unless (integerp tree-attributes-count)
      (error 'type-error
             :expected-type 'integer
             :datum tree-attributes-count))
    (unless (< 0 tree-attributes-count array-total-size-limit)
      (error 'cl-ds:argument-value-out-of-bounds
             :value tree-attributes-count
             :bounds `(< 0 :tree-attributes-count
                         ,array-total-size-limit)
             :argument :tree-attributes-count))
    (unless (typep tree-parameters
                   'cl-grf.tp:fundamental-training-parameters)
      (error 'type-error
             :expected-type 'cl-grf.tp:fundamental-training-parameters
             :datum tree-parameters))
    (unless (integerp trees-count)
      (error 'type-error :expected-type 'integer
                         :datum trees-count))
    (unless (< 0 trees-count array-total-size-limit)
      (error 'cl-ds:argument-value-out-of-bounds
             :value trees-count
             :bounds `(< 0 :trees-count
                         ,array-total-size-limit)
             :argument :trees-count))
    (unless (symbolp forest-class)
      (error 'type-error :expected-type 'symbol
                         :datum forest-class))
    (when (and parallel (cl-grf.tp:parallel tree-parameters))
      (error 'cl-ds:incompatible-arguments
             :arguments '(:parallel :tree-parameters)
             :values `(,parallel ,tree-parameters)
             :format-control "You can't request parallel creation of both the forest and the individual trees at the same time."))
    (unless (integerp tree-sample-size)
      (error 'type-error :expected-type 'integer
                         :datum tree-sample-size))
    (unless (< 0 tree-sample-size array-total-size-limit)
      (error 'cl-ds:argument-value-out-of-bounds
             :value tree-sample-size
             :bounds `(< 0 :tree-sample-size
                         ,array-total-size-limit)
             :argument :tree-sample-size))))


(defmethod leafs-for ((forest fundamental-random-forest)
                      data
                      &optional parallel)
  (check-type data cl-grf.data:data-matrix)
  (funcall (if parallel
               #'lparallel:pmap
               #'map)
           'vector
           (lambda (tree features)
             (~>> (cl-grf.data:sample data :attributes features)
                  (cl-grf.tp:leafs-for tree)))
           (trees forest)
           (attributes forest)))


(defmethod predictions-from-leafs ((forest classification-random-forest)
                                   leafs)
  (iterate
    (declare (type fixnum i))
    (with trees-count = (length leafs))
    (with length = (~> leafs first-elt length))
    (with results = (make-array length :initial-element nil))
    (for i from 0 below length)
    (for total-support = (total-support leafs i))
    (iterate
      (for leaf-group in-vector leafs)
      (for leaf = (aref leaf-group i))
      (for predictions = (cl-grf.alg:predictions leaf))
      (for data-points-count = (cl-grf.data:data-points-count predictions))
      (for attributes-count = (cl-grf.data:attributes-count predictions))
      (for result = (ensure (aref results i)
                      (cl-grf.data:make-data-matrix data-points-count
                                                    attributes-count)))
      (for support = (cl-grf.alg:support leaf))
      (iterate
        (for k from 0 below (array-total-size predictions))
        (incf (row-major-aref result k)
              (/ (row-major-aref predictions k)
                 support))))
    (for result = (aref results i))
    (iterate
      (for k from 0 below (array-total-size result))
      (setf (row-major-aref result k)
            (/ (row-major-aref result k)
               trees-count)))
    (finally (return results))))


(defmethod cl-grf.mp:predict ((random-forest fundamental-random-forest)
                              data)
  (check-type data cl-grf.data:data-matrix)
  (predict random-forest data))


(defmethod cl-grf.mp:make-model ((parameters random-forest-parameters)
                                 train-data
                                 target-data)
  (cl-grf.data:bind-data-matrix-dimensions
      ((train-data-data-points train-data-attributes train-data)
       (target-data-data-points target-data-attributes target-data))
    (let* ((tree-parameters (tree-parameters parameters))
           (trees-count (trees-count parameters))
           (forest-class (forest-class parameters))
           (parallel (parallel parameters))
           (tree-attributes-count (tree-attributes-count parameters))
           (tree-sample-size (tree-sample-size parameters))
           (trees (make-array trees-count))
           (attributes (make-array trees-count)))
      (~>> (selecting-random-indexes tree-attributes-count
                                     train-data-attributes)
           (map-into attributes))
      (funcall (if parallel #'lparallel:pmap-into #'map-into)
               trees
               (lambda (attributes)
                 (let ((data-points (select-random-indexes
                                     tree-sample-size
                                     train-data-data-points)))
                   (cl-grf.mp:make-model tree-parameters
                                         (cl-grf.data:sample
                                          train-data
                                          :data-points data-points
                                          :attributes attributes)
                                         (cl-grf.data:sample
                                          target-data
                                          :data-points data-points))))
               attributes)
      (make forest-class
            :trees trees
            :target-attributes-count target-data-attributes
            :attributes attributes))))
