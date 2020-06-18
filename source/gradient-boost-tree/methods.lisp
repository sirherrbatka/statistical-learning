(cl:in-package #:statistical-learning.gradient-boost-tree)



(defmethod sl.mp:make-training-state ((parameters fundamental-gradient-boost-tree-parameters)
                                      train-data
                                      target-data
                                      &rest initargs
                                      &key attributes weights response expected-value shrinkage
                                      &allow-other-keys)
  (declare (ignore initargs))
  (sl.mp:make-training-state (implementation parameters
                                             :shrinkage shrinkage
                                             :expected-value expected-value)
                             train-data
                             (if (null response)
                                 (target parameters target-data expected-value)
                                 response)
                             :attributes attributes
                             :weights weights))


(defmethod sl.mp:make-model* ((parameters gradient-boosting-implementation)
                              state)
  (declare (ignore train-data target-data))
  (let ((parameters (~> state sl.mp:training-parameters)))
    (make 'gradient-boost-model
          :parameters (gradient-parameters parameters)
          :shrinkage (shrinkage parameters)
          :expected-value (expected-value parameters)
          :root (sl.tp:root (call-next-method)))))


(defmethod sl.mp:make-model* ((parameters fundamental-gradient-boost-tree-parameters)
                              state)
  (sl.mp:make-model* (sl.mp:training-parameters state)
                     state))


(defmethod sl.tp:contribute-predictions* ((parameters regression)
                                          model
                                          data
                                          state
                                          parallel)
  (sl.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (when (null state)
      (setf state (contributed-predictions parameters model data-points-count)))
    (let* ((sums (sl.tp:sums state))
           (shrinkage (shrinkage model))
           (root (sl.tp:root model)))
      (funcall (if parallel #'lparallel:pmap #'map)
               nil
               (lambda (data-point)
                 (let* ((leaf (sl.tp:leaf-for root data data-point))
                        (predictions (sl.tp:predictions leaf)))
                   (incf (sl.data:mref sums data-point 0)
                         (* shrinkage predictions))))
               (sl.tp:indexes state)))
    (incf (sl.tp:contributions-count state))
    state))


(defmethod sl.tp:contribute-predictions* ((parameters classification)
                                          model
                                          data
                                          state
                                          parallel)
  (sl.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (when (null state)
      (setf state (contributed-predictions parameters model data-points-count)))
    (let* ((sums (sl.tp:sums state))
           (number-of-classes (sl.opt:number-of-classes parameters))
           (shrinkage (shrinkage model))
           (root (sl.tp:root model)))
      (funcall (if parallel #'lparallel:pmap #'map)
               nil
               (lambda (data-point)
                 (iterate
                   (declare (type fixnum j))
                   (with leaf = (sl.tp:leaf-for root data data-point))
                   (with predictions = (sl.tp:predictions leaf))
                   (for j from 0 below number-of-classes)
                   (for gradient = (sl.data:mref predictions 0 j))
                   (incf (sl.data:mref sums data-point j)
                         (* shrinkage gradient))))
               (sl.tp:indexes state)))
    (incf (sl.tp:contributions-count state))
    state))


(defmethod sl.tp:extract-predictions* ((parameters regression)
                                       (state sl.tp:contributed-predictions))
  (sl.tp:sums state))


(defmethod sl.tp:extract-predictions* ((parameters classification)
                                       (state sl.tp:contributed-predictions))
  (iterate
    (declare (type fixnum i number-of-classes)
             (type double-float maximum sum)
             (type sl.data:data-matrix sums result))
    (with optimized-function = (optimized-function parameters))
    (with number-of-classes = (sl.opt:number-of-classes optimized-function))
    (with sums = (sl.tp:sums state))
    (with result = (sl.data:make-data-matrix-like sums))
    (for i from 0 below (sl.data:data-points-count sums))
    (for maximum = most-negative-double-float)
    (for sum = 0.0d0)
    (iterate
      (declare (type fixnum j))
      (for j from 0 below number-of-classes)
      (maxf maximum (sl.data:mref sums i j)))
    (iterate
      (declare (type fixnum j)
               (type double-float out))
      (for j from 0 below number-of-classes)
      (for out = (exp (- (sl.data:mref sums i j) maximum)))
      (setf (sl.data:mref result i j) out)
      (incf sum out))
    (iterate
      (declare (type fixnum j))
      (for j from 0 below number-of-classes)
      (setf #1=(sl.data:mref result i j) (/ #1# sum)))
    (finally (return result))))


(defmethod implementation ((parameters classification) &rest initargs)
  (apply #'make 'classification-implementation
         :gradient-parameters parameters
         :maximal-depth (sl.tp:maximal-depth parameters)
         :minimal-difference (sl.tp:minimal-difference parameters)
         :minimal-size (sl.tp:minimal-size parameters)
         :trials-count (sl.tp:trials-count parameters)
         :parallel (sl.tp:parallel parameters)
         initargs))


(defmethod implementation ((parameters regression) &rest initargs)
  (apply #'make 'regression-implementation
         :gradient-parameters parameters
         :maximal-depth (sl.tp:maximal-depth parameters)
         :minimal-difference (sl.tp:minimal-difference parameters)
         :minimal-size (sl.tp:minimal-size parameters)
         :trials-count (sl.tp:trials-count parameters)
         :parallel (sl.tp:parallel parameters)
         initargs))


(defmethod target ((parameters classification) target-data expected-value)
  (iterate
    (with optimized-function = (optimized-function parameters))
    (with number-of-classes = (sl.opt:number-of-classes optimized-function))
    (with data-points-count = (sl.data:data-points-count target-data))
    (with result = (sl.data:make-data-matrix
                    data-points-count
                    number-of-classes))
    (for i from 0 below data-points-count)
    (for target = (truncate (sl.data:mref target-data i 0)))
    (iterate
      (for j from 0 below number-of-classes)
      (setf (sl.data:mref result i j)
            (- (if (= target j) 1 0)
               (sl.data:mref expected-value 0 j))))
    (finally (return result))))


(defmethod target ((parameters regression) target-data expected-value)
  (sl.data:map-data-matrix (lambda (x)
                             (- x expected-value))
                           target-data))


(defmethod sl.tp:initialize-leaf ((training-parameters classification-implementation)
                                  training-state
                                  leaf)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((target-data (sl.mp:target-data training-state))
         (score (sl.tp:loss training-state))
         (data-points-count (sl.data:data-points-count target-data)))
    (declare (type fixnum data-points-count))
    (setf (sl.tp:support leaf) data-points-count
          (sl.tp:predictions leaf) (~>> (sl.data:reduce-data-points #'+ target-data)
                                        (sl.data:map-data-matrix
                                         (lambda (x) (/ x data-points-count))))
          (sl.tp:loss leaf) score)))


(defmethod calculate-expected-value ((parameters classification)
                                     data)
  (iterate
    (with result = (~>> parameters
                        optimized-function
                        sl.opt:number-of-classes
                        (sl.data:make-data-matrix 1)))
    (for i from 0 below (sl.data:data-points-count data))
    (iterate
      (for j from 0 below (sl.data:attributes-count data))
      (incf (sl.data:mref result 0
                          (truncate (sl.data:mref data i 0)))))
    (finally
     (iterate
       (for j from 0 below (sl.data:attributes-count data))
       (for avg = (/ #1=(sl.data:mref result 0 j)
                     (sl.data:data-points-count data)))
       (setf #1# avg))
     (return result))))


(defmethod calculate-expected-value ((parameters regression) data)
  (~> data cl-ds.utils:unfold-table mean))


(defmethod calculate-response ((parameters regression)
                               gathered-predictions
                               expected)
  (~>> (sl.tp:extract-predictions gathered-predictions)
       (sl.opt:response (optimized-function parameters)
                        expected)))


(defmethod calculate-response ((parameters classification)
                               gathered-predictions
                               expected)
  (~>> (sl.tp:sums gathered-predictions)
       (sl.opt:response (optimized-function parameters)
                        expected)))


(defmethod sl.opt:number-of-classes ((object classification))
  (~> object optimized-function sl.opt:number-of-classes))


(defmethod contributed-predictions ((parameters regression)
                                    model
                                    data-points-count)
  (make 'sl.tp:contributed-predictions
        :indexes (sl.data:iota-vector data-points-count)
        :training-parameters parameters
        :sums (sl.data:make-data-matrix data-points-count
                                        1
                                        (expected-value model))))


(defmethod contributed-predictions ((parameters classification)
                                    model
                                    data-points-count)
  (make 'sl.tp:contributed-predictions
        :indexes (sl.data:iota-vector data-points-count)
        :training-parameters parameters
        :sums (iterate
                (with number-of-classes = (sl.opt:number-of-classes parameters))
                (with result = (sl.data:make-data-matrix data-points-count
                                                         number-of-classes))
                (with expected-value = (expected-value model))
                (for i from 0 below data-points-count)
                (iterate
                  (for j from 0 below number-of-classes)
                  (setf (sl.data:mref result i j) (sl.data:mref expected-value 0 j)))
                (finally (return result)))))
