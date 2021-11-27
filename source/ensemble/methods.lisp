(cl:in-package #:statistical-learning.ensemble)


(defmethod initialize-instance :after
    ((instance ensemble)
     &rest initargs)
  (declare (ignore initargs))
  (let* ((trees-count (trees-count instance))
         (tree-batch-size (tree-batch-size instance))
         (tree-attributes-count (tree-attributes-count instance))
         (tree-sample-rate (tree-sample-rate instance))
         (tree-parameters (tree-parameters instance)))
    (unless (< 0.0 tree-sample-rate 1.0)
      (error 'cl-ds:argument-value-out-of-bounds
             :value tree-sample-rate
             :bounds '(< 0 tree-sample-rate 1.0)
             :argument :tree-sample-rate))
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
                   'statistical-learning.mp:fundamental-model-parameters)
      (error 'type-error
             :expected-type 'statistical-learning.mp:fundamental-model-parameters
             :datum tree-parameters))
    (unless (integerp trees-count)
      (error 'type-error :expected-type 'integer
                         :datum trees-count))
    (unless (integerp (/ trees-count tree-batch-size))
      (error 'cl-ds:incompatible-arguments
             :format-control ":TREES-COUNT is supposed to be multiple of :TREE-BATCH-SIZE"
             :parameters '(:tree-batch-size :trees-count)
             :values (list tree-batch-size trees-count)))
    (unless (< 0 trees-count array-total-size-limit)
      (error 'cl-ds:argument-value-out-of-bounds
             :value trees-count
             :bounds `(< 0 :trees-count
                         ,array-total-size-limit)
             :parameter :trees-count))))


(defmethod initialize-instance :after ((instance gradient-boost-ensemble)
                                       &rest initargs)
  (declare (ignore initargs))
  (let ((shrinkage (shrinkage instance)))
    (check-type shrinkage positive-double-float)))


(defmethod statistical-learning.mp:predict ((model ensemble-model)
                                            data
                                            &optional parallel)
  (check-type data statistical-learning.data:data-matrix)
  (let* ((trees (trees model))
         (parameters (statistical-learning.mp:parameters model))
         (tree-parameters (tree-parameters parameters))
         (result (trees-predict model tree-parameters trees data parallel)))
    result))


(defmethod cl-ds.utils:cloning-information append
    ((state gradient-boost-ensemble-state-mixin))
  '((:gradients gradients)))


(defmethod cl-ds.utils:cloning-information append
    ((state ensemble-state))
  '((:all-args all-args)
    (:trees trees)
    (:attributes attributes)
    (:samples samples)
    (:trees-view trees-view)
    (:attributes-view attributes-view)
    (:samples-view samples-view)
    (:indexes indexes)))


(defmethod cl-ds.utils:cloning-information append
    ((state supervised-ensemble-state))
  '((:target-data sl.mp:target-data)
    (:sampling-weights sampling-weights)
    (:assigned-leafs assigned-leafs)
    (:leafs-assigned-p leafs-assigned-p)
    (:weights-calculator-state weights-calculator-state)
    (:weights weights)))


(defmethod cl-ds.utils:cloning-information append
    ((state isolation-forest-ensemble-state))
  '((:c sl.if:c)))


(defmethod update-weights ((calculator static-weights-calculator)
                           tree-parameters
                           ensemble-state
                           ensemble-model)
  nil)


(defmethod update-weights ((calculator dynamic-weights-calculator)
                           (tree-parameters sl.perf:classification)
                           ensemble-state
                           ensemble-model)
  (let* ((parallel (~> ensemble-state sl.mp:parameters parallel))
         (indexes (indexes ensemble-state))
         (target-data (sl.mp:target-data ensemble-state))
         (prev-trees (trees-view ensemble-state))
         (counts (~> ensemble-state weights-calculator-state counts))
         (assigned-leafs (assigned-leafs ensemble-state))
         (weights (sl.mp:weights ensemble-state)))
    (declare (type (simple-array fixnum (* *)) counts)
             (type sl.data:double-float-data-matrix target-data weights))
    (assign-leafs ensemble-state ensemble-model)
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (index)
               (declare (type fixnum index)
                        (optimize (speed 3)))
               (iterate
                 (declare (type double-float expected)
                          (type vector leafs)
                          (type vector assigned-leafs)
                          (type fixnum index)
                          (ignorable tree))
                 (with expected = (sl.data:mref (the sl.data:double-float-data-matrix target-data)
                                                index
                                                0))
                 (with leafs = (aref assigned-leafs index))
                 (for i from (~> leafs length 1-) downto 0)
                 (for tree in-vector prev-trees)
                 (for leaf = (aref leafs i))
                 (incf (aref counts index 0))
                 (for predictions = (sl.tp:predictions leaf))
                 (for prediction =
                      (iterate
                        (declare (type fixnum i))
                        (for i from 0
                             below (sl.data:attributes-count predictions))
                        (finding i maximizing
                                 (sl.data:mref predictions 0 i))))
                 (when (= prediction expected)
                   (incf (aref counts index 1)))))
             indexes)
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (index &aux (total (aref counts index 0)))
               (declare (type fixnum index)
                        (type fixnum total))
               (unless (zerop total)
                 (setf (sl.data:mref (the sl.data:double-float-data-matrix weights)
                                     index 0)
                       (+ (- 1.0d0 (/ (the fixnum (aref counts index 1))
                                      total))
                          double-float-epsilon))))
             indexes)))


(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters random-forest)
                                            &rest initargs
                                            &key train-data target-data weights)
  (bind ((trees-count (trees-count parameters))
         (attributes (make-array trees-count))
         (trees (make-array trees-count))
         (data-points-count (sl.data:data-points-count train-data))
         (assigned-leafs (map-into (make-array data-points-count)
                                   #'vect))
         (indexes (sl.data:iota-vector data-points-count))
         (samples (make-array trees-count)))
    (make 'random-forest-state
          :train-data train-data
          :indexes indexes
          :assigned-leafs assigned-leafs
          :parameters parameters
          :trees trees
          :attributes attributes
          :samples samples
          :target-data target-data
          :weights weights
          :training-parameters parameters
          :all-args initargs)))


(defmethod data-point-samples ((sampler weights-based-data-points-sampler)
                               count
                               state
                               tree-sample-size
                               data-points-count)
  (if-let ((weights (sl.mp:weights state)))
    (map-into (make-array count)
              (curry #'weighted-sample
                     tree-sample-size
                     (sl.random:discrete-distribution weights)))
    (map-into (make-array count) (curry #'sl.data:select-random-indexes
                                        tree-sample-size
                                        data-points-count))))


(defmethod make-weights-calculator-state ((weights-calculator fundamental-weights-calculator)
                                          ensemle-state)
  nil)


(defmethod make-weights-calculator-state ((weights-calculator dynamic-weights-calculator)
                                          ensemble-state)
  (let* ((weights (sl.mp:weights ensemble-state))
         (data-points-count (sl.data:data-points-count weights)))
    (make 'dynamic-weights-calculator-state
          :counts (make-array `(,data-points-count 2)
                              :element-type 'fixnum
                              :initial-element 0))))


(defmethod sl.mp:make-model*/proxy (parameters/proxy
                                    (parameters random-forest)
                                    state)
  (bind ((train-data (sl.mp:train-data state))
         (target-data (sl.mp:target-data state))
         (train-data-attributes (sl.data:attributes-count train-data))
         (tree-batch-size (tree-batch-size parameters))
         (tree-parameters (tree-parameters parameters))
         (trees-count (trees-count parameters))
         (tree-attributes-count (tree-attributes-count parameters))
         ((:accessors trees samples attributes attributes-view
                      samples-view trees-view (weights sl.mp:weights))
          state)
         (attributes-generator (sl.data:selecting-random-indexes
                                tree-attributes-count
                                train-data-attributes))
         (attributes-count (sl.data:attributes-count target-data))
         ((:flet array-view (array &key (from 0) (to trees-count)))
          (make-array (- (min trees-count to) from)
                      :displaced-index-offset (min trees-count from)
                      :displaced-to array))
         (model (make 'random-forest-model
                      :trees trees
                      :parameters parameters
                      :target-attributes-count attributes-count)))
    (statistical-learning.data:bind-data-matrix-dimensions
        ((train-data-data-points train-data-attributes train-data))
      (setf weights (if (null weights)
                        (sl.data:make-data-matrix train-data-data-points
                                                  1
                                                  1.0d0)
                        (copy-array weights)))
      (setf (weights-calculator-state state)
            (~> parameters
                weights-calculator
                (make-weights-calculator-state state)))
      (cl-progress-bar:with-progress-bar (trees-count "Fitting random forest of ~a trees." trees-count)
        (iterate
          (with index = 0)
          (with prev-index = 0)
          (with state-initargs = '())
          (setf prev-index index)
          (while (< index trees-count))
          (setf trees-view (array-view trees
                                        :from index
                                        :to (+ index tree-batch-size)))
          (setf attributes-view (array-view attributes
                                             :from index
                                             :to (+ index tree-batch-size)))
          (map-into attributes-view attributes-generator)
          (setf samples-view (array-view samples
                                         :from index
                                         :to (+ index tree-batch-size)))
          (fit-tree-batch state-initargs state)
          (setf (leafs-assigned-p state) nil)
          (after-tree-fitting parameters tree-parameters state)
          (setf index (min trees-count (+ index tree-batch-size)))
          (for new-trees = (- index prev-index))
          (unless (zerop new-trees)
            (update-weights (weights-calculator parameters)
                            tree-parameters
                            state
                            model))))
      model)))


(defmethod sl.mp:make-model*/proxy (parameters/proxy
                                    (parameters isolation-forest)
                                    state)
  (declare (optimize (debug 3)))
  (bind ((train-data (sl.mp:train-data state))
         (train-data-attributes (sl.data:attributes-count train-data))
         (tree-batch-size (tree-batch-size parameters))
         (tree-parameters (tree-parameters parameters))
         (trees-count (trees-count parameters))
         (tree-attributes-count (tree-attributes-count parameters))
         ((:accessors trees samples attributes attributes-view
                      samples-view trees-view)
          state)
         (attributes-generator (sl.data:selecting-random-indexes
                                tree-attributes-count
                                train-data-attributes))
         ((:flet array-view (array &key (from 0) (to trees-count)))
          (make-array (- (min trees-count to) from)
                      :displaced-index-offset (min trees-count from)
                      :displaced-to array)))
    (statistical-learning.data:bind-data-matrix-dimensions
        ((train-data-data-points train-data-attributes train-data))
      (cl-progress-bar:with-progress-bar (trees-count "Fitting isolation forest of ~a trees."
                                                      trees-count)
        (iterate
          (with index = 0)
          (with state-initargs = '())
          (while (< index trees-count))
          (setf trees-view (array-view trees
                                       :from index
                                       :to (+ index tree-batch-size)))
          (setf attributes-view (array-view attributes
                                             :from index
                                             :to (+ index tree-batch-size)))
          (map-into attributes-view attributes-generator)
          (setf samples-view (array-view samples
                                         :from index
                                         :to (+ index tree-batch-size)))
          (fit-tree-batch state-initargs state)
          (after-tree-fitting parameters tree-parameters state)
          (setf index (min trees-count (+ index tree-batch-size)))))
      (make 'isolation-forest-model
            :trees trees
            :parameters parameters))))

(defmethod sl.mp:weights ((object isolation-forest-ensemble-state))
  nil)

(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters isolation-forest)
                                            &rest initargs
                                            &key data)
  (bind ((trees-count (trees-count parameters))
         (tree-sample-rate (tree-sample-rate parameters))
         (data-points-count (sl.data:data-points-count data))
         (tree-sample-size (* data-points-count tree-sample-rate))
         (indexes (sl.data:iota-vector data-points-count))
         (attributes (make-array trees-count))
         (trees (make-array trees-count))
         (samples (make-array trees-count))
         (c (sl.if:c-factor tree-sample-size)))
    (make 'isolation-forest-ensemble-state
          :all-args `(,@initargs :c ,c)
          :parameters parameters
          :train-data data
          :indexes indexes
          :trees trees
          :attributes attributes
          :samples samples
          :training-parameters parameters)))


(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters gradient-boost-ensemble)
                                            &rest initargs
                                            &key target-data train-data weights)
  (let* ((tree-parameters (tree-parameters parameters))
         (expected-value (sl.gbt:calculate-expected-value
                          tree-parameters
                          target-data))
         (trees-count (trees-count parameters))
         (data-points-count (sl.data:data-points-count train-data))
         (indexes (sl.data:iota-vector data-points-count))
         (assigned-leafs (map-into (make-array data-points-count)
                                   #'vect))
         (attributes (make-array trees-count))
         (trees (make-array trees-count))
         (samples (make-array trees-count)))
    (make 'supervised-gradient-boost-ensemble-state
          :all-args `(,@initargs :expected-value ,expected-value)
          :parameters parameters
          :train-data train-data
          :indexes indexes
          :trees trees
          :assigned-leafs assigned-leafs
          :attributes attributes
          :samples samples
          :target-data target-data
          :weights weights
          :training-parameters parameters)))


(defmethod sl.mp:make-model*/proxy (parameters/proxy
                                    (parameters gradient-boost-ensemble)
                                    state)
  (bind ((train-data (sl.mp:train-data state))
         (target-data (sl.mp:target-data state))
         (train-data-attributes (sl.data:attributes-count train-data))
         (target-data-attributes (sl.data:attributes-count target-data))
         (tree-batch-size (tree-batch-size parameters))
         (tree-parameters (tree-parameters parameters))
         (trees-count (trees-count parameters))
         (parallel (parallel parameters))
         (tree-attributes-count (tree-attributes-count parameters))
         ((:accessors trees samples attributes attributes-view
                      samples-view trees-view
                      (response gradients)
                      (weights sl.mp:weights))
          state)
         (attributes-generator (sl.data:selecting-random-indexes
                                tree-attributes-count
                                train-data-attributes))
         ((:flet array-view (array &key (from 0) (to trees-count)))
          (make-array (- (min trees-count to) from)
                      :displaced-index-offset (min trees-count from)
                      :displaced-to array))
         (model (make 'gradient-boost-ensemble-model
                      :trees trees
                      :parameters parameters
                      :target-attributes-count target-data-attributes)))
    (statistical-learning.data:bind-data-matrix-dimensions
        ((train-data-data-points train-data-attributes train-data))
      (setf weights (if (null weights)
                        (sl.data:make-data-matrix train-data-data-points
                                                  1
                                                  1.0d0)
                        (copy-array weights)))
  (cl-progress-bar:with-progress-bar (trees-count "Fitting gradient boost ensemble of ~a trees." trees-count)
    (iterate
      (with contributed = nil)
      (with shrinkage = (shrinkage parameters))
      (for index
           from 0
           below trees-count
           by tree-batch-size)
      (while (< index trees-count))
      (setf trees-view (array-view trees
                                   :from index
                                   :to (+ index tree-batch-size)))
      (setf attributes-view (array-view attributes
                                        :from index
                                        :to (+ index tree-batch-size)))
      (map-into attributes-view attributes-generator)
      (setf samples-view (array-view samples
                                     :from index
                                     :to (+ index tree-batch-size)))
      (fit-tree-batch `(:response ,response :shrinkage ,shrinkage)
                      state)
      (for new-contributed = (contribute-trees model
                                               tree-parameters
                                               trees-view
                                               train-data
                                               parallel
                                               contributed))
      (setf response (sl.gbt:calculate-response tree-parameters
                                                new-contributed
                                                target-data)
            contributed new-contributed))))
    model))


(defmethod sl.perf:performance-metric* ((parameters ensemble)
                                        type
                                        target
                                        predictions
                                        weights)
  (sl.perf:performance-metric* (tree-parameters parameters)
                               type
                               target
                               predictions
                               weights))


(defmethod sl.perf:average-performance-metric ((parameters ensemble)
                                               metrics
                                               &key type)
  (sl.perf:average-performance-metric (tree-parameters parameters)
                                      metrics
                                      :type type))


(defmethod sl.perf:errors ((parameters ensemble)
                           target
                           predictions)
  (sl.perf:errors (tree-parameters parameters)
                  target
                  predictions))


(defmethod sl.perf:default-performance-metric ((parameters ensemble))
  (~> parameters tree-parameters sl.perf:default-performance-metric))


(defmethod leafs ((ensemble ensemble-model)
                  data
                  &optional parallel)
  (sl.data:check-data-points data)
  (let* ((data-points (sl.data:data-points-count data))
         (parameters (sl.mp:parameters ensemble))
         (tree-parameters (tree-parameters parameters))
         (splitter (sl.tp:splitter tree-parameters))
         (trees (trees ensemble))
         (roots (map 'vector #'sl.tp:root trees))
         (result (sl.data:make-data-matrix data-points
                                           1
                                           nil
                                           t))
         (unfolded (cl-ds.utils:unfold-table result)))
    (iterate
      (for i from 0 below (length unfolded))
      (setf (aref unfolded i) i))
    (funcall (if parallel #'lparallel:pmap-into #'map-into)
             unfolded
             (lambda (index)
               (iterate
                 (with result = (~> trees length make-array))
                 (for i from 0)
                 (for root in-vector roots)
                 (for leaf = (sl.tp:leaf-for splitter
                                             root
                                             data
                                             index
                                             ensemble))
                 (setf (aref result i) leaf)
                 (finally (return result))))
             unfolded)
    result))


(defmethod make-tree-training-state/proxy
    (ensemble-parameters/proxy
     tree-parameters/proxy
     (ensemble-parameters ensemble)
     tree-parameters
     ensemble-state
     attributes
     data-points
     initargs)
  (apply #'sl.mp:make-training-state
         tree-parameters
         :data-points data-points
         :attributes attributes
         initargs))


(defmethod after-tree-fitting/proxy
    (ensemble-parameters/proxy
     tree-parameters/proxy
     ensemble-parameters
     tree-parameters
     ensemble-state)
  nil)


(defmethod assign-leafs ((state supervised-ensemble-state) model)
  (when (leafs-assigned-p state)
    (return-from assign-leafs nil))
  (bind ((parallel (~> state sl.mp:parameters parallel))
         (parameters (sl.mp:parameters state))
         (tree-parameters (tree-parameters parameters))
         (splitter (sl.tp:splitter tree-parameters))
         (assigned-leafs (assigned-leafs state))
         (train-data (sl.mp:train-data state))
         (trees (trees-view state))
         (indexes (indexes state)))
    (map nil #'sl.tp:force-tree trees)
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (index)
               (declare (type fixnum index))
               (iterate
                 (with leafs = (aref assigned-leafs index))
                 (for tree in-vector trees)
                 (for leaf = (sl.tp:leaf-for splitter
                                             (sl.tp:root tree)
                                             train-data
                                             index
                                             model))
                 (vector-push-extend leaf leafs)))
             indexes)
    (setf (leafs-assigned-p state) t)
    nil))
