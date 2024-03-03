(cl:in-package #:statistical-learning.ensemble)


(defmethod initialize-instance :after
    ((instance weights-based-data-points-sampler)
     &rest initargs)
  (declare (ignore initargs))
  (let ((sampling-rate (sampling-rate instance)))
    (check-type sampling-rate real)
    (unless (< 0.0 sampling-rate 1.0)
      (error 'cl-ds:argument-value-out-of-bounds
             :value sampling-rate
             :bounds '(< 0 sampling-rate 1.0)
             :argument :sampling-rate))))


(defmethod initialize-instance :after
    ((instance gradient-based-one-side-sampler)
     &rest initargs)
  (declare (ignore initargs))
  (let ((large-gradient-sampling-rate (large-gradient-sampling-rate instance))
        (small-gradient-sampling-rate (small-gradient-sampling-rate instance)))
    (check-type large-gradient-sampling-rate real)
    (check-type small-gradient-sampling-rate real)
    (unless (< 0.0 large-gradient-sampling-rate 1.0)
      (error 'cl-ds:argument-value-out-of-bounds
             :value large-gradient-sampling-rate
             :bounds '(< 0 large-gradient-sampling-rate 1.0)
             :argument :large-gradient-sampling-rate))
    (unless (< 0.0 small-gradient-sampling-rate 1.0)
      (error 'cl-ds:argument-value-out-of-bounds
             :value small-gradient-sampling-rate
             :bounds '(< 0 small-gradient-sampling-rate 1.0)
             :argument :small-gradient-sampling-rate))))


(defmethod initialize-instance :after
    ((instance ensemble)
     &rest initargs)
  (declare (ignore initargs))
  (let* ((trees-count (trees-count instance))
         (tree-batch-size (tree-batch-size instance))
         (tree-attributes-count (tree-attributes-count instance))
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
  (let* ((trees (trees model))
         (result (trees-predict model trees (sl.data:wrap data) parallel)))
    (if (arrayp result) result (sl.data:data result))))


(defmethod cl-ds.utils:cloning-information append
    ((state gradient-boost-ensemble-state-mixin))
  '((:gradients gradients)))


(defmethod cl-ds.utils:cloning-information append
    ((state ensemble-state))
  '((:all-args all-args)
    (:parameters sl.mp:parameters)
    (:trees trees)
    (:attributes attributes)
    (:samples samples)
    (:trees-view trees-view)
    (:attributes-view attributes-view)
    (:samples-view samples-view)
    (:sampler-state sampler-state)
    (:train-data sl.mp:train-data)))


(defmethod cl-ds.utils:cloning-information append
    ((state supervised-ensemble-state))
  '((:target-data sl.mp:target-data)
    (:assigned-leafs assigned-leafs)
    (:leafs-assigned-p leafs-assigned-p)))


(defmethod cl-ds.utils:cloning-information append
    ((state random-forest-state))
  '((:weights sl.mp:weights)
    (:weights-calculator-state weights-calculator-state)))


(defmethod cl-ds.utils:cloning-information append
    ((state isolation-forest-ensemble-state))
  '((:c sl.if:c)))


(defmethod cl-ds.utils:cloning-information append
    ((model ensemble-model))
  '((:trees trees)))


(defmethod cl-ds.utils:cloning-information append
    ((model random-forest-model))
  '((:target-attributes-count target-attributes-count)))


(defmethod cl-ds.utils:cloning-information append
    ((model gradient-boost-ensemble-model))
  '((:target-attributes-count target-attributes-count)))


(defmethod update-weights ((calculator static-weights-calculator)
                           tree-parameters
                           ensemble-state
                           ensemble-model)
  nil)


(defmethod update-weights ((calculator dynamic-weights-calculator)
                           (tree-parameters sl.perf:regression)
                           ensemble-state
                           ensemble-model)
  (assign-leafs ensemble-state ensemble-model)
  (bind ((parallel (~> ensemble-state sl.mp:parameters parallel))
         (target-data (sl.mp:target-data ensemble-state))
         (prev-trees (trees-view ensemble-state))
         (assigned-leafs (assigned-leafs ensemble-state))
         (weights (sl.mp:weights ensemble-state)))
    (declare (type sl.data:double-float-data-matrix target-data))
    (regression-errors weights
                       prev-trees
                       target-data
                       assigned-leafs
                       parallel)))


(defmethod update-weights ((calculator dynamic-weights-calculator)
                           (tree-parameters sl.perf:classification)
                           ensemble-state
                           ensemble-model)
  (let* ((parallel (~> ensemble-state sl.mp:parameters parallel))
         (target-data (sl.mp:target-data ensemble-state))
         (prev-trees (trees-view ensemble-state))
         (counts (~> ensemble-state weights-calculator-state counts))
         (assigned-leafs (assigned-leafs ensemble-state))
         (weights (sl.mp:weights ensemble-state)))
    (declare (type (simple-array single-float (* *)) counts)
             (type sl.data:double-float-data-matrix target-data))
    (assign-leafs ensemble-state ensemble-model)
    (sl.data:data-matrix-map (lambda (index data)
                               (declare (type fixnum index))
                               (iterate
                                 (declare (type double-float expected)
                                          (type vector leafs)
                                          (type vector assigned-leafs)
                                          (type fixnum index)
                                          (ignorable tree))
                                 (with expected = (aref data index 0))
                                 (with leafs = (aref assigned-leafs index))
                                 (for i from (~> leafs length 1-) downto 0)
                                 (for tree in-vector prev-trees)
                                 (for leaf = (aref leafs i))
                                 (incf (aref counts index 0))
                                 (if (vectorp leaf)
                                     (iterate
                                       (with length = (length leaf))
                                       (for l in-vector leaf)
                                       (for predictions = (sl.tp:predictions l))
                                       (for prediction =
                                            (iterate
                                              (declare (type fixnum i))
                                              (for i from 0
                                                   below (array-dimension predictions 1))
                                              (finding i maximizing (aref predictions 0 i))))
                                       (when (= prediction expected)
                                         (incf (aref counts index 1) (/ 1 length))))
                                     (bind ((predictions (sl.tp:predictions leaf))
                                            (prediction
                                             (iterate
                                               (declare (type fixnum i))
                                               (for i from 0
                                                    below (array-dimension predictions 1))
                                               (finding i maximizing
                                                        (aref predictions 0 i)))))
                                       (when (= prediction expected)
                                         (incf (aref counts index 1)))))))
                             target-data
                             parallel)
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (index &aux (total (aref counts index 0)))
               (declare (type fixnum index)
                        (type single-float total))
               (unless (zerop total)
                 (setf (sl.data:mref weights index 0)
                       (+ (- 1.0d0 (/ (the single-float (aref counts index 1))
                                      total))
                          double-float-epsilon))))
             (sl.data:index target-data))))


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
         (samples (make-array trees-count)))
    (make 'random-forest-state
          :train-data train-data
          :assigned-leafs assigned-leafs
          :parameters parameters
          :trees trees
          :attributes attributes
          :samples samples
          :target-data target-data
          :weights weights
          :training-parameters parameters
          :all-args initargs)))


(defmethod data-points-samples ((sampler gradient-based-one-side-sampler)
                                state
                                count)
  (if-let ((response (gradients state)))
    (bind ((data-points-count (sl.data:data-points-count response))
           (attributes-count (sl.data:attributes-count response))
           ((:flet response-at-point (point))
            (declare (optimize (speed 3) (safety 0)))
            (iterate
              (declare (type fixnum i)
                       (type double-float result))
              (with result = 0.0d0)
              (for i from 0 below attributes-count)
              (incf result (sl.data:mref (the sl.data:double-float-data-matrix response)
                                         point i))
              (finally (return result))))
           ((:flet >gradient (point-a point-b))
            (declare (optimize (speed 3) (safety 0)))
            (> (response-at-point point-a)
               (response-at-point point-b)))
           (small-gradient-sampling-rate (small-gradient-sampling-rate sampler))
           (large-gradient-sampling-rate (large-gradient-sampling-rate sampler))
           (ordered-data-points (~> (sl.data:iota-vector data-points-count)
                                    (sort #'>gradient)))
           (large-gradient-count (min (ceiling (* large-gradient-sampling-rate data-points-count))
                                      data-points-count))
           (small-gradient-count (min (floor (* small-gradient-sampling-rate data-points-count))
                                      (- data-points-count large-gradient-count)))
           (total-count (+ large-gradient-count small-gradient-count))
           ;; (normalization (/ (- 1.0d0 large-gradient-sampling-rate) small-gradient-sampling-rate))
           ((:flet generate-sample (&aux (r (make-array total-count
                                                        :element-type 'fixnum))))
            (declare (optimize (speed 3) (safety 0) (debug 0))
                     (type (simple-array fixnum (*)) r))
            (replace r ordered-data-points
                     :start1 0
                     :start2 0
                     :end1 large-gradient-count
                     :end2 large-gradient-count)
            (~>> (sl.data:select-random-indexes small-gradient-count
                                                (- data-points-count
                                                   large-gradient-count))
                 (cl-ds.utils:transform
                  (lambda (i) (declare (type fixnum i))
                    (let ((offset (the fixnum (+ large-gradient-count i))))
                      (aref ordered-data-points offset))))
                 (the (simple-array fixnum (*)) _)
                 (replace r _
                          :start1 large-gradient-count
                          :start2 0
                          :end1 total-count))
            r))
      (declare (type fixnum total-count large-gradient-count))
      ;; (iterate
      ;;   (for i from large-gradient-count below data-points-count)
      ;;   (for point = (aref ordered-data-points i))
      ;;   (iterate
      ;;     (for j from 0 below (sl.data:attributes-count response))
      ;;     (setf #1=(sl.data:mref response point j)
      ;;           (* normalization #1#))))
      (funcall (if (~> state sl.mp:parameters parallel)
                   #'lparallel:pmap-into
                   #'map-into)
               (make-array count)
               #'generate-sample))
    (~>> state
         sl.mp:train-data
         sl.data:data-points-count
         (curry #'sl.data:select-random-indexes
                (max (small-gradient-sampling-rate sampler)
                     (large-gradient-sampling-rate sampler)))
         (map-into (make-array count)))))


(defmethod data-points-sampler ((parameters isolation-forest))
  (make-instance 'weights-based-data-points-sampler
                 :sampling-rate (tree-sample-rate parameters)))


(defmethod data-points-samples ((sampler weights-based-data-points-sampler)
                                state
                                count)
  (let* ((train-data (sl.mp:train-data state))
         (data-points-count (sl.data:data-points-count train-data))
         (sampling-rate (sampling-rate sampler))
         (tree-sample-size (floor (* sampling-rate data-points-count))))
    (if-let ((weights (sl.mp:weights state)))
      (map-into (make-array count)
                (curry #'weighted-sample
                       tree-sample-size
                       (sl.random:discrete-distribution weights)))
      (map-into (make-array count) (curry #'sl.data:select-random-indexes
                                          tree-sample-size
                                          data-points-count)))))


(defmethod make-weights-calculator-state ((weights-calculator fundamental-weights-calculator)
                                          ensemle-state)
  nil)


(defmethod make-weights-calculator-state ((weights-calculator dynamic-weights-calculator)
                                          ensemble-state)
  (let* ((weights (sl.mp:weights ensemble-state))
         (data-points-count (sl.data:data-points-count weights)))
    (make 'dynamic-weights-calculator-state
          :counts (make-array `(,data-points-count 2)
                              :element-type 'single-float
                              :initial-element 0.0))))


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
      (~> (prune-trees (pruning parameters)
                       model
                       train-data
                       target-data)
          (refine-trees (refinement parameters)
                        _
                        train-data
                        target-data)))))


(defmethod sl.mp:make-model*/proxy (parameters/proxy
                                    (parameters isolation-forest)
                                    state)
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
         (attributes (make-array trees-count))
         (trees (make-array trees-count))
         (samples (make-array trees-count))
         (c (sl.if:c-factor tree-sample-size)))
    (make 'isolation-forest-ensemble-state
          :all-args `(,@initargs :c ,c)
          :parameters parameters
          :train-data data
          :trees trees
          :attributes attributes
          :samples samples
          :training-parameters parameters)))


(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters gradient-boost-ensemble)
                                            &rest initargs
                                            &key target-data train-data)
  (let* ((tree-parameters (tree-parameters parameters))
         (expected-value (sl.gbt:calculate-expected-value
                          tree-parameters
                          target-data))
         (trees-count (trees-count parameters))
         (data-points-count (sl.data:data-points-count train-data))
         (assigned-leafs (map-into (make-array data-points-count)
                                   #'vect))
         (attributes (make-array trees-count))
         (trees (make-array trees-count))
         (samples (make-array trees-count)))
    (make 'supervised-gradient-boost-ensemble-state
          :all-args `(,@initargs :expected-value ,expected-value)
          :parameters parameters
          :train-data train-data
          :trees trees
          :assigned-leafs assigned-leafs
          :attributes attributes
          :samples samples
          :target-data target-data
          :training-parameters parameters)))


(defmethod sl.mp:make-model*/proxy :before
    (parameters/proxy (parameters ensemble-model) state)
  (setf (sampler-state state)
        (~> parameters
            weights-calculator
            (make-data-points-sampler-state state))))


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
                      (response gradients))
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
              contributed new-contributed)))
    (prune-trees (pruning parameters)
                 model
                 train-data
                 target-data)))


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
         (unfolded (~> result sl.data:data cl-ds.utils:unfold-table)))
    (iterate
      (for i from 0 below (length unfolded))
      (setf (aref unfolded i) i))
    (funcall (if parallel #'lparallel:pmap-into #'map-into)
             unfolded
             (lambda (index)
               (iterate
                 (with result = (~> trees length make-array))
                 (for i from 0)
                 (for tree in-vector trees)
                 (for root in-vector roots)
                 (for leaf = (sl.tp:leaf-for splitter
                                             root
                                             data
                                             index
                                             tree))
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
  (declare (optimize (debug 3)))
  (when (leafs-assigned-p state)
    (return-from assign-leafs nil))
  (bind ((parallel (~> state sl.mp:parameters parallel))
         (parameters (sl.mp:parameters state))
         (tree-parameters (tree-parameters parameters))
         (splitter (sl.tp:splitter tree-parameters))
         (assigned-leafs (assigned-leafs state))
         (train-data (sl.mp:train-data state))
         (trees (trees-view state)))
    (check-type train-data sl.data:data-matrix)
    (map nil #'sl.tp:force-tree trees)
    (sl.data:data-matrix-map-data-points
     (lambda (index train-data)
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
                             train-data
                             parallel)
    (setf (leafs-assigned-p state) t)
    nil))


(defmethod sl.mp:weights ((state gradient-boost-ensemble-state-mixin))
  nil)


(defmethod prune-trees ((algorithm (eql nil))
                        ensemble
                        train-data
                        target-data)
  ensemble)


(defmethod refine-trees ((algorithm (eql nil))
                          ensemble
                          train-data
                          target-data)
  ensemble)
